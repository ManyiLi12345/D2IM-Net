import numpy as np
import torch
import torch.nn as nn
import torch.utils.data


# D2IM setting
from models.Nets import ResEncoder as D2IM_ImageEncoder
from models.Nets import ImnetDecoder as D2IM_IMDecoder
from models.Nets import ResDecoder as D2IM_DetailDecoder

class D2IM_Net(nn.Module):
    def __init__(self, config):
        super(D2IM_Net, self).__init__()
        # network parameters
        self.img_res = config.img_res
        self.model_dir = config.model_dir
        self.exp_name = config.exp_name
        self.iscuda = config.cuda

        # network modules
        if(config.exp_name=='imnet'):
            self.image_encoder = IMNET_ImageEncoder(self.model_dir)
            self.detail_decoder = IMNET_DetailDecoder()
            self.im_decoder = IMNET_IMDecoder()
        elif(config.exp_name=='disn'):
            self.image_encoder = DISN_ImageEncoder(self.model_dir)
            self.detail_decoder = DISN_DetailDecoder()
            self.im_decoder = DISN_IMDecoder()
        elif(config.exp_name=='d2im'):
            self.image_encoder = D2IM_ImageEncoder(self.model_dir)
            self.detail_decoder = D2IM_DetailDecoder()
            self.im_decoder = D2IM_IMDecoder()
        else:
            print("base mode error!")
            exit()
        self.mseloss = nn.MSELoss()
        self.l1loss = nn.L1Loss()
        
        # conv kernel for computing the gradients on 2D
        kernel_nx = [[0,0,0],[-1,1,0],[0,0,0]]
        kernel_ny = [[0,0,0],[0,1,0],[0,-1,0]]
        self.normal_weight = torch.tensor([kernel_nx,kernel_ny]).float()
        self.normal_weight = self.normal_weight.unsqueeze(1)
        self.cam_zvec = torch.tensor(np.array([[0.0,0.0,1.0]])).float()
        if(self.iscuda):
            self.normal_weight = self.normal_weight.cuda()
            self.cam_zvec = self.cam_zvec.cuda()

    """ Transform direction vectors from worldview to camview"""
    def project_vector_to_camview(self, vecs, transmat):
        plus = torch.zeros((vecs.size(0),vecs.size(1),1)).cuda()
        homovecs = torch.cat([vecs,plus], dim=2)
        homovecs = torch.matmul(homovecs, transmat)
        return homovecs[:,:,:3]

    """ Transform direction vectors from camview to worldview"""
    def project_vector_to_worldview(self, vecs, transmat):
        plus = torch.tensor([[[0.0], [0.0], [0.0], [1.0]]]).cuda()
        transmat = torch.cat([transmat, plus], dim=2)
        inv_transmat = torch.inverse(transmat)
        plus = torch.zeros((vecs.size(0),vecs.size(1),1)).cuda()
        homovecs = torch.cat([vecs,plus], dim=2)
        homovecs = torch.matmul(homovecs, inv_transmat)
        return homovecs[:,:,:3]

    """ Project 3D points to 2D pixels """
    def project_points_to_pixels(self, points, transmat):
        plus = torch.ones((points.size(0),points.size(1),1)).cuda()
        homopoints = torch.cat([points,plus], dim=2)
        
        homopoints = torch.matmul(homopoints, transmat)
        homopoints[:,:,0] = torch.div(homopoints[:,:,0],homopoints[:,:,2])
        homopoints[:,:,1] = torch.div(homopoints[:,:,1],homopoints[:,:,2])

        gt_pixels = homopoints[:,:,:2] 
        gt_uv = gt_pixels*2.0/self.img_res - 1.0 # uv in range [-1,1]
        gt_depth = homopoints[:,:,2]
        
        gt_pixels = gt_pixels.long()
        gtpixels_outside = gt_pixels < 0
        gt_pixels = gt_pixels.masked_fill(gtpixels_outside, 0)
        gtpixels_outside = gt_pixels>=self.img_res
        gt_pixels = gt_pixels.masked_fill(gtpixels_outside, self.img_res-1)
        return gt_uv, gt_pixels, gt_depth

    """ From 2D feature maps to per-point features based on uv coordinates """
    """ "gt_uv" in range [-1,1] """
    def project_featmap_by_uv(self, gt_uv, featmap_list):
        feat_list = []
        for featmap in featmap_list:
            feats = nn.functional.grid_sample(featmap, gt_uv.unsqueeze(2), mode='bilinear', padding_mode="border")
            feats = feats[:,:,:,0]
            feat_list.append(feats)
        pointfeats = torch.cat(feat_list, dim=1)
        pointfeats = pointfeats.permute(0,2,1)
        return pointfeats

    """ From 2D feature maps to per-point features based on pixel coordinates """
    """ gt_pixels in range [0,img_res] """
    def project_featmap_by_px(self, gt_pixels, featmap):
        C = featmap.size(1)
        featmap_res = featmap.size(2)
        featmap = featmap.permute((0,2,3,1))
        featmap = featmap.view(featmap.size(0), -1, C)

        pointfeats = []
        for i in range(featmap.size(0)):
            gtpixels_per_shape = gt_pixels[i,:,1] * featmap_res + gt_pixels[i,:,0]
            point_feats_per_shape = torch.index_select(featmap[i,:,:], 0, gtpixels_per_shape)
            pointfeats.append(point_feats_per_shape.unsqueeze(0))
        pointfeats = torch.cat(pointfeats, dim=0)
        return pointfeats

    def forward(self, gt_points, gt_values, gt_gradients, mc_image, gt_transmat, scale):
        # for checking the projection
        # gt_uv, _, _ = self.project_points_to_pixels(points, transmat)
        # check_projection(gt_uv, gt_points, gt_values, transmat, mc_image, self.img_res)
        #exit()

        # image encoder
        gt_rgba = mc_image[:,:3,:,:]
        gt_normal = mc_image[:,3:,:,:]
        featvecs, featmap_list = self.image_encoder(gt_rgba)

        # compute the 2D-3D correspondense
        gt_uv, gt_pixels, gt_depth = self.project_points_to_pixels(gt_points, gt_transmat)

        # two decoder branches 
        if(self.exp_name == "imnet"):
            base_values = self.im_decoder(featvecs, gt_points)
        elif(self.exp_name == "disn"):
            pointfeats = self.project_featmap_by_uv(gt_uv, featmap_list)
            global_values, local_values = self.im_decoder(featvecs, pointfeats, gt_points)
            base_values = global_values + local_values
        elif(self.exp_name == 'd2im'):
            base_values = self.im_decoder(featvecs, gt_points)
        pred_disp = self.detail_decoder(featmap_list)

        # classify the "front-side" and "back-side" points
        projected_gradients = self.project_vector_to_camview(gt_gradients, gt_transmat)
        front_weights = 0.5-torch.sign(projected_gradients[:,:,2].unsqueeze(2))*0.5
        # project the per-point displacements
        pred_disp_front = pred_disp[:,0,:,:].unsqueeze(1)
        pred_disp_back = pred_disp[:,1,:,:].unsqueeze(1)
        pred_point_disp_front = self.project_featmap_by_px(gt_pixels, pred_disp_front)
        pred_point_disp_back = self.project_featmap_by_px(gt_pixels, pred_disp_back)
        
        # base_loss for coarse shapes
        loss_base = self.mseloss(base_values, gt_values)

        # sdf_loss for final reconstruction
        #pred_sdf_values = base_values + front_weights*pred_point_disp_front + (1-front_weights)*pred_point_disp_back
        #loss_sdf = self.l1loss(pred_sdf_values, gt_values)
        front_values = base_values + pred_point_disp_front
        front_weights = front_weights
        loss_front = torch.sum(front_weights * torch.abs(front_values - gt_values))/torch.sum(front_weights)
        # back
        back_values = base_values + pred_point_disp_back
        back_weights = 1-front_weights
        loss_back = torch.sum(back_weights * torch.abs(back_values - gt_values))/torch.sum(back_weights)
        loss_sdf = (loss_front+loss_back)*0.5

        # laplacian_loss
        pred_normal_map = nn.functional.conv2d(pred_disp_front, self.normal_weight, padding=1)
        pred_nx_map = pred_normal_map[:,0,:,:].unsqueeze(1)
        pred_ny_map = pred_normal_map[:,1,:,:].unsqueeze(1)
        pred_nx2_map = nn.functional.conv2d(pred_nx_map, self.normal_weight, padding=1)
        pred_nx2_map = pred_nx2_map[:,0,:,:].unsqueeze(1)
        pred_ny2_map = nn.functional.conv2d(pred_ny_map, self.normal_weight, padding=1)
        pred_ny2_map = pred_ny2_map[:,1,:,:].unsqueeze(1)
        pred_n2_map = (pred_nx2_map+pred_ny2_map)/2.0 # laplacian on the 2D displacement

        gt_nx_map = gt_normal[:,2,:,:].unsqueeze(1)
        gt_ny_map = gt_normal[:,1,:,:].unsqueeze(1)
        gt_nx2_map = nn.functional.conv2d(gt_nx_map, self.normal_weight, padding=1)
        gt_nx2_map = gt_nx2_map[:,0,:,:].unsqueeze(1)
        gt_ny2_map = nn.functional.conv2d(gt_ny_map, self.normal_weight, padding=1)
        gt_ny2_map = gt_ny2_map[:,1,:,:].unsqueeze(1)
        gt_n2_map = (gt_nx2_map+gt_ny2_map)/2.0 # gradient on the 2D normal map
        
        gt_point_n2 = self.project_featmap_by_px(gt_pixels, gt_n2_map)
        # projection from "gradient w.r.t. 3D point" to "gradient w.r.t. 2D pixel"
        #gt_point_n2 = gt_point_n2*(2.0*gt_depth.unsqueeze(2))/(49.0*scale)
        pred_point_n2 = self.project_featmap_by_px(gt_pixels, pred_n2_map)
        pred_point_n2 = pred_point_n2*49.0*scale/(2.0*gt_depth.unsqueeze(2))
        # "+" because the displacement should be opposite to the surface geometry
        lap_weights = (gt_values<0.1).float()*front_weights
        loss_laplacian = torch.sum(lap_weights * (pred_point_n2 + gt_point_n2)**2)/torch.sum(lap_weights)
        
        return loss_base, loss_sdf, loss_laplacian

    def inference_sdf_with_detail(self, points, rgba_image, transmat):
        # compute the 3D-2D correspondence
        gt_uv, gt_pixels, _ = self.project_points_to_pixels(points, transmat)

        featvecs, featmap_list = self.image_encoder(rgba_image)
        pred_disp = self.detail_decoder(featmap_list)
        if(self.exp_name == "imnet"):
            base_values = self.im_decoder(featvecs, points)
        elif(self.exp_name == "disn"):
            pointfeats = self.project_featmap_by_uv(gt_uv, featmap_list)
            global_values, local_values = self.im_decoder(featvecs, pointfeats, points)
            base_values = global_values + local_values
        elif(self.exp_name == 'd2im'):
            base_values = self.im_decoder(featvecs, points)

        # compute delta
        vecs = self.cam_zvec.unsqueeze(0)
        vecs = self.project_vector_to_worldview(vecs, transmat)
        vecs = nn.functional.normalize(vecs, p=2, dim=2)*0.03
        vecs = vecs.repeat(points.size(0),points.size(1),1)
        
        # estimate the frontness
        delta_points = points + vecs
        delta_gt_uv, _, _ = self.project_points_to_pixels(delta_points, transmat)
        if(self.exp_name == "imnet"):
            delta_base_values = self.im_decoder(featvecs, delta_points)
        elif(self.exp_name == "disn"):
            delta_pointfeats = self.project_featmap_by_uv(delta_gt_uv, featmap_list)
            delta_global_values, delta_local_values = self.im_decoder(featvecs, delta_pointfeats, delta_points)
            delta_base_values = delta_global_values + delta_local_values
        elif(self.exp_name == 'd2im'):
            delta_base_values = self.im_decoder(featvecs, delta_points)
            
        front_weights = 0.5-0.5*torch.sign(delta_base_values - base_values)

        pred_disp_front = pred_disp[:,0,:,:].unsqueeze(1)
        pred_disp_back = pred_disp[:,1,:,:].unsqueeze(1)
        pred_point_disp_front = self.project_featmap_by_px(gt_pixels, pred_disp_front)
        pred_point_disp_back = self.project_featmap_by_px(gt_pixels, pred_disp_back)
        final_values = base_values + pred_point_disp_front*front_weights + pred_point_disp_back*(1-front_weights)
        
        return final_values

    def inference_normal(self, rgba_image):
        featvecs, featmap_list = self.image_encoder(rgba_image)
        pred_disp = self.detail_decoder(featmap_list)

        pred_disp = pred_disp[:,0,:,:].unsqueeze(1)
        pred_normal_map = nn.functional.conv2d(pred_disp, self.normal_weight, padding=1)
        pred_nx_map = pred_normal_map[:,0,:,:].unsqueeze(1)
        pred_ny_map = pred_normal_map[:,1,:,:].unsqueeze(1)
        pred_nx2_map = nn.functional.conv2d(pred_nx_map, self.normal_weight, padding=1)
        pred_nx2_map = pred_nx2_map[:,0,:,:].unsqueeze(1)
        pred_ny2_map = nn.functional.conv2d(pred_ny_map, self.normal_weight, padding=1)
        pred_ny2_map = pred_ny2_map[:,1,:,:].unsqueeze(1)
        pred_n2_map = (pred_nx2_map+pred_ny2_map)/2.0
        return pred_disp, pred_nx_map, pred_ny_map, pred_n2_map

    
    
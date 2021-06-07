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
        self.exp_name = config.exp_name
        self.img_res = config.img_res
        self.model_dir = config.model_dir
        self.iscuda = config.cuda

        # network modules
        if(config.exp_name=='d2im'):
            self.image_encoder = D2IM_ImageEncoder(self.model_dir)
            self.detail_decoder = D2IM_DetailDecoder()
            self.im_decoder = D2IM_IMDecoder()
        else:
            print("exp_name error!")
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
        camvecs = torch.cat([vecs,plus], dim=2)
        camvecs = torch.matmul(camvecs, transmat)
        return camvecs[:,:,:3]

    """ Transform direction vectors from camview to worldview"""
    def project_vector_to_worldview(self, vecs, transmat):
        plus = torch.tensor([[[0.0], [0.0], [0.0], [1.0]]]).cuda()
        transmat = torch.cat([transmat, plus], dim=2)
        inv_transmat = torch.inverse(transmat)

        plus = torch.zeros((vecs.size(0),vecs.size(1),1)).cuda()
        worldvecs = torch.cat([vecs,plus], dim=2)
        worldvecs = torch.matmul(worldvecs, inv_transmat)
        return worldvecs[:,:,:3]

    """ Project 3D points to 2D pixels """
    def project_points_to_pixels(self, points, transmat):
        plus = torch.ones((points.size(0),points.size(1),1)).cuda()
        homopoints = torch.cat([points,plus], dim=2)
        
        campoints = torch.matmul(homopoints, transmat)
        campoints[:,:,0] = torch.div(campoints[:,:,0],campoints[:,:,2])
        campoints[:,:,1] = torch.div(campoints[:,:,1],campoints[:,:,2])

        pixels = campoints[:,:,:2] # in range [0,img_res]
        uv = pixels*2.0/self.img_res - 1.0 # in range [-1,1]
        depth = campoints[:,:,2]
        
        # rounding the pixel indices
        pixels = pixels.long()
        pixels_outside = pixels < 0
        pixels = pixels.masked_fill(pixels_outside, 0)
        pixels_outside = pixels>=self.img_res
        pixels = pixels.masked_fill(pixels_outside, self.img_res-1)
        return uv, pixels, depth

    """ From 2D feature maps to per-point features """
    """ "uv" in range [-1,1] """
    def project_featmap_by_uv(self, uv, featmap_list):
        feat_list = []
        for featmap in featmap_list:
            feats = nn.functional.grid_sample(featmap, uv.unsqueeze(2), mode='bilinear', padding_mode="border")
            feats = feats[:,:,:,0]
            feat_list.append(feats)
        pointfeats = torch.cat(feat_list, dim=1)
        pointfeats = pointfeats.permute(0,2,1)
        return pointfeats

    """ From 2D feature maps to per-point features based on pixel coordinates """
    """ pixels in range [0,img_res] """
    '''
    def project_featmap_by_px(self, pixels, featmap):
        C = featmap.size(1)
        featmap_res = featmap.size(2)
        featmap = featmap.permute((0,2,3,1))
        featmap = featmap.view(featmap.size(0), -1, C)

        pointfeats = []
        for i in range(featmap.size(0)):
            pixels_per_shape = pixels[i,:,1] * featmap_res + pixels[i,:,0]
            point_feats_per_shape = torch.index_select(featmap[i,:,:], 0, pixels_per_shape)
            pointfeats.append(point_feats_per_shape.unsqueeze(0))
        pointfeats = torch.cat(pointfeats, dim=0)
        return pointfeats
    '''

    def forward(self, points, values, gradients, mc_image, transmat, scale):
        uv, pixels, depth = self.project_points_to_pixels(points, transmat)

        # image encoder
        rgb_image = mc_image[:,:3,:,:]
        normal_image = mc_image[:,3:,:,:]
        featvecs, featmap_list = self.image_encoder(rgb_image)

        if(self.exp_name == 'd2im'):
            base_values = self.im_decoder(featvecs, points)
        else:
            print("exp_name error!")
            exit()

        pred_disp = self.detail_decoder(featmap_list)

        # classify the "front-side" and "back-side" points
        # based on the SDF gradient direction in the camera's view
        cam_gradients = self.project_vector_to_camview(gradients, transmat)
        front_weights = 0.5-torch.sign(cam_gradients[:,:,2].unsqueeze(2))*0.5
        
        # project the per-point displacements
        pred_front_disp = pred_disp[:,0,:,:].unsqueeze(1) # front disp. map
        pred_back_disp = pred_disp[:,1,:,:].unsqueeze(1) # back disp. map
        pred_point_front_disp = self.project_featmap_by_uv(uv, [pred_front_disp])
        pred_point_back_disp = self.project_featmap_by_uv(uv, [pred_back_disp])
        
        # base_loss for coarse shapes
        loss_base = self.mseloss(base_values, values)

        # sdf_loss
        front_values = base_values + pred_point_front_disp
        back_values = base_values + pred_point_back_disp
        back_weights = 1-front_weights
        loss_front = torch.sum(front_weights * torch.abs(front_values - values))/torch.sum(front_weights)        
        loss_back = torch.sum(back_weights * torch.abs(back_values - values))/torch.sum(back_weights)
        loss_sdf = (loss_front+loss_back)*0.5

        # laplacian_loss
        pred_normal_map = nn.functional.conv2d(pred_front_disp, self.normal_weight, padding=1)
        pred_nx_map = pred_normal_map[:,0,:,:].unsqueeze(1)
        pred_ny_map = pred_normal_map[:,1,:,:].unsqueeze(1)
        pred_nx2_map = nn.functional.conv2d(pred_nx_map, self.normal_weight, padding=1)
        pred_nx2_map = pred_nx2_map[:,0,:,:].unsqueeze(1)
        pred_ny2_map = nn.functional.conv2d(pred_ny_map, self.normal_weight, padding=1)
        pred_ny2_map = pred_ny2_map[:,1,:,:].unsqueeze(1)
        pred_n2_map = (pred_nx2_map+pred_ny2_map)/2.0 # 2D disp. Laplacian

        gt_nx_map = normal_image[:,2,:,:].unsqueeze(1)
        gt_ny_map = normal_image[:,1,:,:].unsqueeze(1)
        gt_nx2_map = nn.functional.conv2d(gt_nx_map, self.normal_weight, padding=1)
        gt_nx2_map = gt_nx2_map[:,0,:,:].unsqueeze(1)
        gt_ny2_map = nn.functional.conv2d(gt_ny_map, self.normal_weight, padding=1)
        gt_ny2_map = gt_ny2_map[:,1,:,:].unsqueeze(1)
        gt_n2_map = (gt_nx2_map+gt_ny2_map)/2.0 # Laplacian of 3D SDF
        
        gt_Laplacian = self.project_featmap_by_uv(uv, [gt_n2_map])
        pred_Laplacian = self.project_featmap_by_uv(uv, [pred_n2_map])
        # Note that the gt_Laplacian and pred_Laplacian are not matched so far.
        # One is "Laplacian w.r.t. 3D coordinates" and the other is "Laplacian w.r.t. 2D coordinates"
        # Next we need to project the Laplacian to 3D. 
        # Note that "Laplacian computation first and then projection" is equavalent to 
        # "Projection first and then Laplacian"
        # Also, instead of projecting the gt_Laplacian, we operate on the pred_Laplacian, just to obtain a larger loss value.
        pred_Laplacian = pred_Laplacian*49.0*scale/(2.0*depth.unsqueeze(2))
        lap_weights = (values<0.1).float()*front_weights # only consider the points near the front surface
        loss_laplacian = torch.sum(lap_weights * (pred_Laplacian + gt_Laplacian)**2)/torch.sum(lap_weights)
        # pred"+"gt because the GT SDF has negative values inside the object. So the GT normal is opposite to this SDF gradients.
        return loss_base, loss_sdf, loss_laplacian

    def inference_sdf_with_detail(self, points, rgb_image, transmat):
        uv, pixels, _ = self.project_points_to_pixels(points, transmat)
        featvecs, featmap_list = self.image_encoder(rgb_image)
        pred_disp = self.detail_decoder(featmap_list)
        if(self.exp_name == 'd2im'):
            base_values = self.im_decoder(featvecs, points)
        else:
            print("exp_name error!")
            exit()

        # estimate the gradients
        vecs = self.cam_zvec.unsqueeze(0)
        vecs = self.project_vector_to_worldview(vecs, transmat)
        vecs = nn.functional.normalize(vecs, p=2, dim=2)*0.03
        vecs = vecs.repeat(points.size(0),points.size(1),1)
        
        # estimate the frontness
        delta_points = points + vecs
        delta_uv, _, _ = self.project_points_to_pixels(delta_points, transmat)
        if(self.exp_name == 'd2im'):
            delta_base_values = self.im_decoder(featvecs, delta_points)
            
        front_weights = 0.5-0.5*torch.sign(delta_base_values - base_values)

        pred_front_disp = pred_disp[:,0,:,:].unsqueeze(1)
        pred_back_disp = pred_disp[:,1,:,:].unsqueeze(1)
        pred_point_front_disp = self.project_featmap_by_uv(uv, [pred_front_disp])
        pred_point_back_disp = self.project_featmap_by_uv(uv, [pred_back_disp])
        final_values = base_values + pred_point_front_disp*front_weights + pred_point_back_disp*(1-front_weights)
        return final_values

import os
import cv2
import mcubes
import torch
import numpy as np
from argparse import ArgumentParser

def get_args():
    parser = ArgumentParser(description='D2IM-Net')
    parser.add_argument('--cuda', type=bool, default=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--plot_every_batch', type=int, default=10)
    parser.add_argument('--save_every_epoch', type=int, default=20)
    parser.add_argument('--test_every_epoch', type=int, default=20)
    parser.add_argument('--load_pretrain', type=bool, default=True)

    parser.add_argument('--viewnum', type=int, default=36)
    parser.add_argument('--img_res', type=int, default=224)
    parser.add_argument('--mcube_znum', type=int, default=128)
    parser.add_argument('--test_pointnum', type=int, default=100000)

    parser.add_argument('--lr', type=float, default=0.00005)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--cam_batch_size', type=int, default=16)
    parser.add_argument('--cam_lr', type=float, default=0.00005)
    parser.add_argument('--train_batch_size', type=int, default=16)
    parser.add_argument('--test_batch_size', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--sampling_mode', type=str, default='weighted')
    parser.add_argument('--exp_name', type=str, default='d2im')
    
    parser.add_argument('--data_dir', default='./data/')    
    parser.add_argument('--model_dir', default='./ckpt/models')
    parser.add_argument('--output_dir', default='./ckpt/outputs')
    parser.add_argument('--log', default='log.txt')
    
    # some selected chairs with details
    testlist = [
        {'cat_id':'03001627', 'shape_id':'f1f40596ca140cc89cfc48dba5c0e481', 'cam_id':32},
        {'cat_id':'03001627', 'shape_id':'cbcc5cd1aec9f3413aa677469bbdd68c', 'cam_id':32},
        {'cat_id':'03001627', 'shape_id':'9b4d530487df4aa94b3c42e318f3affc', 'cam_id':32},
        {'cat_id':'03001627', 'shape_id':'8ec79ed07c19aa5cfebad4f49b26ec52', 'cam_id':32},
        {'cat_id':'03001627', 'shape_id':'cd939609247df917d9d3572bbd9cf789', 'cam_id':32},
        {'cat_id':'03001627', 'shape_id':'3df44d30265f697e7e684d25d4dcaf0', 'cam_id':32},
        {'cat_id':'03001627', 'shape_id':'103d77d63f0d68a044e6721e9db29c1b', 'cam_id':32},
        {'cat_id':'03001627', 'shape_id':'cce9ffdcc7ca8ddea300840c9d7bfa74', 'cam_id':32},
        {'cat_id':'03001627', 'shape_id':'d1436d73b3060f2ffd6176b35397ccd5', 'cam_id':32},
        {'cat_id':'03001627', 'shape_id':'d4e0707b680e61e0593ebeeedbff73b', 'cam_id':32},
        {'cat_id':'03001627', 'shape_id':'d4e974b1d57693ab65ae658fdfdd758d', 'cam_id':32},
        {'cat_id':'03001627', 'shape_id':'d794f296dbe579101e046801e2748f1a', 'cam_id':32},
        {'cat_id':'03001627', 'shape_id':'d7e26a070ee3b35cdf6cfab91d65bb91', 'cam_id':32},
        {'cat_id':'03001627', 'shape_id':'d8f3c4bf9266150a579147ba03140821', 'cam_id':32},
        {'cat_id':'03001627', 'shape_id':'d93133f1f5be7da1191c3762b497eca9', 'cam_id':32},
        {'cat_id':'03001627', 'shape_id':'d9346fab44b6308f40ef1c8b63a628f9', 'cam_id':32},
        {'cat_id':'03001627', 'shape_id':'db812fdfacf4db8df51f77a6d7299806', 'cam_id':32},
        {'cat_id':'03001627', 'shape_id':'dee24ea8622e2005dd0e1ff930f92c75', 'cam_id':32},
        {'cat_id':'03001627', 'shape_id':'df23ca11080bb439676c272956dad3c2', 'cam_id':32},
        {'cat_id':'03001627', 'shape_id':'e1897a4391784bc2e8b2b8dc0c816caf', 'cam_id':32},
        {'cat_id':'03001627', 'shape_id':'e26ac9cc4c44f8709531b4e1074af521', 'cam_id':32},
        {'cat_id':'03001627', 'shape_id':'e2809feb8c0b535686c701087a194026', 'cam_id':32},
        {'cat_id':'03001627', 'shape_id':'e5b6a3cf96b36a1613660685f1489e72', 'cam_id':32},
        {'cat_id':'03001627', 'shape_id':'e5f381dd81cca6a36a348179d243c2c5', 'cam_id':32},
        {'cat_id':'03001627', 'shape_id':'e955b63a4bd738857178717457aa5d20', 'cam_id':32},
        {'cat_id':'03001627', 'shape_id':'ea577a1e9dffe7b55096c0dd2594842a', 'cam_id':32},
        {'cat_id':'03001627', 'shape_id':'ebaf425ea2c92f73a3bafec3b56382db', 'cam_id':32},
        {'cat_id':'03001627', 'shape_id':'f083f6f7e9770fb7b161f36d4e309050', 'cam_id':32},
        {'cat_id':'03001627', 'shape_id':'f1f40596ca140cc89cfc48dba5c0e481', 'cam_id':32},
        {'cat_id':'03001627', 'shape_id':'f6096649f87c37f1af7c7ad2549a1b15', 'cam_id':32},
        {'cat_id':'03001627', 'shape_id':'fc14c1aa7831f4cbcaef18b2fd3fb17c', 'cam_id':32},
        {'cat_id':'03001627', 'shape_id':'fc97c771d058556f593de14e4664635a', 'cam_id':32}
    ]

    args = parser.parse_args()
    args.testlist = testlist
    args.catlist = ['03001627']#, '02691156', '02828884', '02933112', '03211117', '03636649', '03691459', '04090263', '04256520', '04379243', '04530566','02958343', '04401088']
    return args

def print_log(log_fname, logline):
    f = open(log_fname,'a')
    f.write(logline)
    f.write('\n')
    f.close()

def save_checkpoint(epoch, model, optimizer, bestloss, output_filename):
    state = {'epoch': epoch + 1,
         'state_dict': model.state_dict(),
         'optimizer': optimizer.state_dict(),
         'bestloss': bestloss}
    torch.save(state, output_filename)

def load_checkpoint(cp_filename, model, optimizer=None):
    checkpoint = torch.load(cp_filename)
    model.load_state_dict(checkpoint['state_dict'])
    if(optimizer is not None):
        optimizer.load_state_dict(checkpoint['optimizer'])
    epoch = checkpoint['epoch']
    if('bestloss' in checkpoint.keys()):
        bestloss = checkpoint['bestloss']
    else:
        bestloss = 10000000
    return epoch, model, optimizer, bestloss

def load_model(cp_filename, model):
    checkpoint = torch.load(cp_filename)
    model.load_state_dict(checkpoint['state_dict'])
    epoch = checkpoint['epoch']
    return epoch, model

""" sample grid points in the 3D space [-0.5,0.5]^3 """
def sample_grid_points(xnum,ynum,znum):
    gridpoints = np.zeros((xnum, ynum, znum, 3))
    for i in range(xnum):
        for j in range(ynum):
            for k in range(znum):
                gridpoints[i, j, k, :] = [i,j,k] 
    gridpoints[:,:,:,0] = (gridpoints[:,:,:,0] + 0.5)/xnum - 0.5
    gridpoints[:,:,:,1] = (gridpoints[:,:,:,1] + 0.5)/ynum - 0.5
    gridpoints[:,:,:,2] = (gridpoints[:,:,:,2] + 0.5)/znum - 0.5
    return gridpoints

""" render the occupancy field to 3 image views """
def render_grid_occupancy(fname, gridvalues, threshold=0):
    signmat = np.sign(gridvalues - threshold)
    img1 = np.clip((np.amax(signmat, axis=0)-np.amin(signmat, axis=0))*256, 0,255).astype(np.uint8)
    img2 = np.clip((np.amax(signmat, axis=1)-np.amin(signmat, axis=1))*256, 0,255).astype(np.uint8)
    img3 = np.clip((np.amax(signmat, axis=2)-np.amin(signmat, axis=2))*256, 0,255).astype(np.uint8)

    fname_without_suffix = fname[:-4]
    cv2.imwrite(fname_without_suffix+'_1.png',img1)
    cv2.imwrite(fname_without_suffix+'_2.png',img2)
    cv2.imwrite(fname_without_suffix+'_3.png',img3)

""" marching cube """
def render_implicits(fname, gridvalues, threshold=0):
    vertices, triangles = mcubes.marching_cubes(-1.0*gridvalues, threshold)
    vertices[:,0] = ((vertices[:,0] + 0.5)/gridvalues.shape[0] - 0.5)
    vertices[:,1] = ((vertices[:,1] + 0.5)/gridvalues.shape[1] - 0.5)
    vertices[:,2] = ((vertices[:,2] + 0.5)/gridvalues.shape[2] - 0.5)    
    write_ply(fname, vertices, triangles)

def write_obj(fname, vertices, triangles):
    fout = open(fname, 'w')
    for ii in range(len(vertices)):
        fout.write("v "+str(vertices[ii,0])+" "+str(vertices[ii,1])+" "+str(vertices[ii,2])+"\n")
    for ii in range(len(triangles)):
        fout.write("f "+str(int(triangles[ii,0])+1)+" "+str(int(triangles[ii,1])+1)+" "+str(int(triangles[ii,2])+1)+"\n")
    fout.close()

def write_ply(fname, vertices, triangles):
	fout = open(fname, 'w')
	fout.write("ply\n")
	fout.write("format ascii 1.0\n")
	fout.write("element vertex "+str(len(vertices))+"\n")
	fout.write("property float x\n")
	fout.write("property float y\n")
	fout.write("property float z\n")
	fout.write("element face "+str(len(triangles))+"\n")
	fout.write("property list uchar int vertex_index\n")
	fout.write("end_header\n")
	for ii in range(len(vertices)):
		fout.write(str(vertices[ii,0])+" "+str(vertices[ii,1])+" "+str(vertices[ii,2])+"\n")
	for ii in range(len(triangles)):
		fout.write("3 "+str(triangles[ii,0])+" "+str(triangles[ii,1])+" "+str(triangles[ii,2])+"\n")
	fout.close()


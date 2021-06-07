import os
import h5py
import random
import numpy as np

import torch
from torch.utils import data 

from datasets import CamUtils
from datasets import DataUtils

class CamDataset(data.Dataset):
    def __init__(self, config, status):
        self.catlist = ['03001627', '02691156', '02828884', '02933112', '03211117', '03636649', '03691459', '04090263', '04256520', '04379243', '04530566','02958343', '04401088']
        self.viewnum = config.viewnum
        self.config = config
        self.world_matrix = np.array([[0, 0, 1.0], [0, 1.0, 0], [-1.0, 0, 0]])
        #self.num_points = 2048

        # read the shape ids from the files
        shape_id_list_from_files = []
        cat_id_list_from_files = []
        for cat_id in self.catlist:
            filename = config.data_dir + cat_id + '_' + status + '.lst'
            shape_ids = DataUtils.read_shape_ids_from_file(filename)
            cat_ids = [cat_id for _ in shape_ids]
            shape_id_list_from_files = shape_id_list_from_files + shape_ids
            cat_id_list_from_files = cat_id_list_from_files + cat_ids

        # check the existence of the data files to form the dataset
        datalist = []
        transmat_list = []
        for i in range(len(shape_id_list_from_files)):
            cat_id = cat_id_list_from_files[i]
            shape_id = shape_id_list_from_files[i]
            # each data needs "rgba_image, normal_image, edge_image"
            rgba_dir = config.image_dir + cat_id + '_easy/' + shape_id + '/'
            cam_fn = config.image_dir + cat_id + '_easy/' + shape_id + '/rendering_metadata.txt'
            h5_fn = config.h5_dir + cat_id + '/' + shape_id + '/data.h5'
            # check the existence of the files
            rgba_existence = os.path.exists(rgba_dir) #We assume all the images (rgba, normal, edge) exist when the rgba_dir exists.
            h5_existence = os.path.exists(h5_fn)
            # add the data into the dataset
            if(rgba_existence and h5_existence):
                data = {'rgba_dir':rgba_dir, 'cat_id':cat_id, 'shape_id':shape_id, 'h5_fn':h5_fn}
                transmats = self.read_transmats(cam_fn)
                transmat_list.append(transmats)
                datalist.append(data)
        
        self.datalist = datalist
        self.transmat_list = transmat_list
        self.datasize = len(self.datalist)
        self.world_matrix = np.array([[0, 0, 1.0], [0, 1.0, 0], [-1.0, 0, 0]])
        
        print('Finished loading the %s dataset: %d data.'%(status, self.datasize))

    
    def read_transmats(self, cam_fn):
        fid = open(cam_fn, 'r')
        line = fid.readline()
        cam_list = []
        RT_list = []
        while(line):
            startid = line.index('[')
            endid = line.index(']')
            data = line[startid+1:endid]            
            data = data.split(',')
            cam = []
            for d in data:
                cam.append(float(d))
            K, RT = CamUtils.getBlenderProj(cam[0], cam[1], cam[3], img_w=224, img_h=224)
            RT = np.transpose(RT)
            RT_list.append(RT)
            cam_list.append(cam)
            line = fid.readline()
        return RT_list

    def __getitem__(self, index):
        # read the points and values
        transmats = self.transmat_list[index]

        data = self.datalist[index]
        cat_id = data['cat_id']
        shape_id = data['shape_id']
        rgba_dir = data['rgba_dir']
        h5_fn = data['h5_fn']

        f = h5py.File(h5_fn, 'r')
        samples = f['pc_sdf_sample']
        cent = f['norm_params'][:3]
        cent = np.expand_dims(cent,axis=0)
        scale = f['norm_params'][3]
        points = samples[:,:3]*scale
        samplings = points + np.repeat(cent, points.shape[0],axis=0)
        
        # read the images 
        rand_cam_id = random.randint(0,self.viewnum-1)
        rgba_image = DataUtils.read_rgba_image(rgba_dir, rand_cam_id)
        transmat = transmats[rand_cam_id]

        # return the data
        samplings = torch.tensor(samplings).float()
        transmat = torch.tensor(transmat).float()
        rgba_image = torch.tensor(rgba_image).float()
        rgba_image = rgba_image.permute(2,0,1)
        rgba_image = rgba_image[:3,:,:]
        return rgba_image, transmat, samplings

    def __len__(self):
        return self.datasize

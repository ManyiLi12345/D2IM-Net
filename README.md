# D2IM-Net: learning detail disentangled implicit fields from single images

## Data preparation
Our training data includes the images, normal maps and the 3D supervision, all downloaded from datasets provided by [DISN](https://github.com/Xharlie/ShapenetRender_more_variation). Please download the dataset into folder ./data/.

The datasets need to be pre-processed for our training.
```
python ./preprocessing/process_normals.py
python ./preprocessing/estimate_density.py # pre-compute the point sampling density for our weighted sampling strategy during training.
python ./preprocessing/estimate_gradient.py # pre-compute the SDF gradients to classify the front-side and back-side nearby points.
```

## Training and testing
For training and testing, please directly run
```
python train.py
```
and 
```
python test.py
```
For the default settings, please refer to utils.py.

During training, it saves the reconstructions of some selected images with GT camera after every a few epochs, in "./ckpt/outputs/".

The test script is to reconstruct from the input image with inferred camera. Before testing, please make sure the trained models, including camera model and D2IM model, are in the './ckpt/models/'. We provide our pre-trained model [here](https://drive.google.com/drive/folders/1UMNDy_NA9bKqe6T_xcTnxRMiea4neWw-?usp=sharing).


## Citation


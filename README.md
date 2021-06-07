# D2IM-Net: learning detail disentangled implicit fields from single images

## Data preparation
Our training data includes the images, normal maps and the 3D supervision, all downloaded from datasets provided by DISN. The datasets need to be pre-processed for our training.

```
python ./preprocessing/process_normals.py
python ./preprocessing/estimate_density.py # pre-compute the point sampling density for our weighted sampling strategy during training.
python ./preprocessing/estimate_gradient.py # pre-compute the SDF gradients to classify the front-side and back-side nearby points.
```

## Training and testing

```
python train.py
```
The default setting is to train our network on the chair category. To train on all the 13 categories, please change the setting in utils.py. During training, it saves the reconstructions of some selected images after every a few epochs, in "./ckpt/outputs/".

```
python test.py
```
The test script is to reconstruct from the input image with inferred camera. Before testing, please make sure the trained models, including camera model and D2IM model, are in the './ckpt/models/'. We provide our pre-trained model here.


## Citation


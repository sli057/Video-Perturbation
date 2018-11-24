# Video-Perturbation
Code for NDSS paper: Stealthy Adversarial Perturbations Against Real-Time Video Classification Systems (https://arxiv.org/pdf/1807.00458.pdf)

## Requirements:

1. Have installed the tensorflow >= 1.2 version
2. You must have installed the following two python libs: 
a) tensorflow 
b) Pillow
c) numpy
d) cv2

3. You must have downloaded the UCF101 (Action Recognition Data Set)
4. Each single avi file is decoded with 5FPS in a single directory. 
	Ref: https://github.com/hx173149/C3D-tensorflow
	- you can use the `./list/convert_video_to_images.sh` script to decode the ucf101 video files
	- run `./list/convert_video_to_images.sh .../UCF101 5`


## Usage:

1. `python train.py` will train the C-DUP or 2D-DUP generator.
The trained model will saved in `./G_model_3D` or `./G_model_2D` directory.
2. `python generate_pertubation.py` will generate perturbations 
from the C-DUP generator and save the perturbations in `./G_model_3D` 
or genearte perturbations from the 2D-DUP generator and save the perturbations in `./G_model_2D` directory.
3. `python test.py` will test the attack success rate with the generated pertubation.


## Trained model:

## Experiment results:

See the paper


#!/bin/sh
#Get the orignial celebA dataset
python get_data.py 1nNGRZOl9X4ryeCkEKMBbxNKeCHOp4ShU ../data/celeba/img_align_celeba.zip
#get the GAN generated fake celebA dataset
python get_data.py 1XzUvl3q1-cf9M6TaXWyIfK0gJV33lfcH ../data/celeba/fake_images.zip

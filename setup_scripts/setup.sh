

#!/bin/sh
#Get the orignial celebA dataset
python get_data.py 1nNGRZOl9X4ryeCkEKMBbxNKeCHOp4ShU ../data/celeba/img_align_celeba.zip
#get the GAN generated fake celebA dataset
python get_data.py 1XzUvl3q1-cf9M6TaXWyIfK0gJV33lfcH ../data/celeba/fake_images.zip
#record.zip file
python get_data.py 12l8MJ8PNCiDCJSoOHezaikm3BN9CgCbk ../FairnessModels/record/record.zip

#models.zip file
python get_data.py 1Ap4zG6YRJukxjbptaaqdVs0b_fkT9b_G ../models.zip

#celeba.zip
python get_data.py 14mfX1QdtEbxpBDrrZmy7FDPKEpufD0Eq ../data/celeba.zip


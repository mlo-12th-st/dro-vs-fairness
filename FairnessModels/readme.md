# Fairness with Latent Space Debiasing

If viewing on the web interface, there are files missing from this repository. They will be downloaded with a script to fetch large files. You should see a "record" folder appear after running the script.

## Paper Overview
Fair Attribute Classification through Latent Space De-biasing  
Authors: Vikram V. Ramaswamy and Sunnie S. Y. Kim and Olga Russakovsky  
IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)
2021

Looked at image classification problem and uses three metrics to compute fairness (derived in 3 other papers, page 5)  
   Then proposed a way to even out the distribution with GANs, does not change original dataset, but augments it with new images
   Then evaluates accuracy/precision/recall and fairness metrics (some of which have accuracy, precision, and recall built in)


## Running the GAN-Debiasing method
In a bash shell run in the current directory,
```
bash run_fairness.sh
```

###
This will do the following:
1. Train classifiers for the target attribute (blond) and protected attribute (male)
2. Generate Latent Vectors for the images in the dataset
3. Generate new images from the latent vectors z
4. Classify each of the newly generated images for the target and protected attribute
5. Generate the Complimentary latent vectors z' and use them to generate images without the inherent biases from the dataset.
6. Evaluate the classifier on the new dataset.



## Notes From Training
### Time Complexity
- On a Nvidia 1660Ti GPU 6 GB, it took approximately 38 minutes per epoch to train the classifier on the entire dataset ~202000 images.
- Generating the scores (predicted classifications) for each of the generated images took less than 10 minutes on the same GPU.
- Generating the Images took about 35 minutes (Generates 175,000 Images)

### Intuitions
- The fairness model is trying to look at correlations between features and separate the features so they are independent. For example, in the dataset, there is a correlation between whether a person is blonde and their gender. So when training a classifier if you are trying to predict gender from a photo, it would look at the hair and say if you are blond then your likely to be female, which is not generally the case. It is an unintentional skew in the dataset so this model tries to correct this skew by using a GAN to generate more images to equalize the underlying distribution between gender and smiling by adding more photos that look like males smiling.

- Also Note that the images generated are using the latent vectors from the original dataset, so many of the images look similar to certain people in the dataset, but there are some images that come out with unrecognizable abnormalities, for example an image where half of the face is cut off, it appears that is was looking at images where a person was turned sideways but we don't know for sure.

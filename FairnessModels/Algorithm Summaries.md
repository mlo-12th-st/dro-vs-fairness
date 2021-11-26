1) Fair Attribute Classification through Latent Space De-biasing  
Looked at image classificaiton problem and uses three metrics to compute fairness (derived in 3 other papers, page 5)  
   Then proposed a way to even out the distribution with GANs, does not change original dataset, but augments it with new images (did something similer with embedding models for text data)  
   Then evaluates accuracy/precision/recall and fairness metrics (some of which have acc/p/r built in)

(Other interesting papars - mostly on text data though)
2) Fairness in Streaming Submodular Maximization: Algorithms and Hardness  
  
Fancy Way of Saying they developed some approximation algorithms to assess the quality of different machine learning methods with and without fairness.   
      
Paper: https://proceedings.neurips.cc//paper/2020/file/9d752cb08ef466fc480fba981cfa44a1-Paper.pdf  
Code: https://github.com/google-research/google-research/tree/master/fair_submodular_maximization_2020  

3) Fairness without Demographics through Adversarially Reweighted Learning  
  
This algorithm predicts fairness when demographic data is not known, showed that by applying fairness metric to learning model that it increased accuracy scores for sparse data groups.  
Paper: https://proceedings.neurips.cc//paper/2020/file/07fc15c9d169ee48573edd749d25945d-Paper.pdf  
Code: https://github.com/google-research/google-research/tree/master/group_agnostic_fairness







Notes on Gan Debiasing Algorithm Training for Presentation:
- The fairness model is trying to look at correlations between features and seperate the features so they are independent. For example, in the dataset, there is a correlation between whether a person is smiling and their gender. So when training a classifier if you are trying to predict gender from a photo, it would look at smiling and say if you are smiling then your likely to be female, which is not generally the case. It is an unintentional skew in the dataset so this model tries to correct this skew by using a GAN to generate more images to equalize the underlying distribution between gender and smiling by adding more photos that look like males smiling. 

- Also Note that the images generated are using the latent vectors from the original dataset, so many of the images look similar to certain people in the dataset, but there are soem images that come out with unrecognizable abnormalities, for example an image where half of the face is cut off,it appears that is was looking at images where a person was turned sideways but we don't know for sure. 

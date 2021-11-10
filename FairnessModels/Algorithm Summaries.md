1) Fair Attribute Classification through Latent Space De-biasing  
Looked at image classificaiton problem and uses three metrics to compute fairness (derived in 3 other papers, page 5)  
   Then proposed a way to even out the distribution with GANs, does not change original dataset, but augments it with new images (did something similer with embedding models for text data)  
   Then evaluates accuracy/precision/recall and fairness metrics (some of which have acc/p/r built in)



(back-burner)
2) Fairness in Streaming Submodular Maximization: Algorithms and Hardness  
  
Fancy Way of Saying they developed some approximation algorithms to assess the quality of different machine learning methods with and without fairness.   
      
Paper: https://proceedings.neurips.cc//paper/2020/file/9d752cb08ef466fc480fba981cfa44a1-Paper.pdf  
Code: https://github.com/google-research/google-research/tree/master/fair_submodular_maximization_2020  

3) Fairness without Demographics through Adversarially Reweighted Learning  
  
This algorithm predicts fairness when demographic data is not known, showed that by applying fairness metric to learning model that it increased accuracy scores for sparse data groups.  
Paper: https://proceedings.neurips.cc//paper/2020/file/07fc15c9d169ee48573edd749d25945d-Paper.pdf  
Code: https://github.com/google-research/google-research/tree/master/group_agnostic_fairness
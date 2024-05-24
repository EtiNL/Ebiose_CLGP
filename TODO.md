# graph encoders
- add an initialisation method for their parameters
- verify the compatibility between them and CLGP
- make them more customizable with config files

# text encoders
- add other architectures like REact to test

# train
- add a weight and biases logging to track trainning of the models
- add a multiple GPU trainning framework
- add an hyperparameter gridsearch

# dataloader
- make a tokenizer for both graph node feature and text

# inference
- make an interface for the ebiose graph generator and the models for it to initialize with a graph and score its graph generation with 
$d(\varphy (Graph), T(Prompt) )$
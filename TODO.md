# graph encoders
- add an initialisation method for their parameters
- verify the compatibility between them and CLGP
- make them more customizable with config files

# text encoders
- add other architectures like REact to test
- add attention mask to bert: 
                              attention_mask = torch.tensor([[1, 1, 1, 1, 1, 1], [1, 0, 0, 0, 0, 0]])
                              output = model(input_ids, attention_mask=attention_mask)

# train
- add an hyperparameter gridsearch
- add test accuracy (for each prompt in test, look at the most similar graph and verify if the selected graph resolves the problem)

# dataloader

# inference
- make an interface for the ebiose graph generator and the models for it to initialize with a graph and score its graph generation with 
$d(\varphi (Graph), T(Prompt) )$
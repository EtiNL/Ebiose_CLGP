# Ebiose Contrastive Language Graph Pair (CLGP)

## Dependencies

```sh
sudo apt-get update
sudo apt-get install libpq-dev
```

Then, install the required Python packages:

```sh
pip install -r requirements.txt
pip install -e .
```

## Overview

### Project Structure

#### `setup.py`
Defines the setup configuration for the project package, including dependencies and package metadata.

#### `train.py`
The main training script includes:
- **Loading configurations**: Sets up the environment and loads the necessary configurations.
- **Training function**: Manages the training loop, logging, gradient clipping, and checkpointing.
- **Saving utilities**: Handles model checkpoints and other utility functions.
- **Main function**: Initializes configurations, datasets, and starts the training process.

#### `utils.py`
Contains utility functions for:
- Directory creation.
- Configuration loading.
- JSON handling.

#### `train_utils.py`
Includes helper functions for:
- Setting up learning rate schedules.
- Logging.
- Ensuring reproducibility by setting random seeds.

#### `test_data_utils.py`
A simple script to test a tokenizer.

#### `search_and_retrieval.py`
Defines functions for connecting to:
- A Neo4j graph database.
- A PostgreSQL database.
- Retrieving graph data from these databases.

#### Model Definitions
- `CLGP.py`: Defines the main model class `CLGP` which includes initialization, parameter settings, and the forward pass.
- `CKGGP.py`: Defines another model class `CKGGP`, a variant of the `CLGP` model.

#### Transformer and Neural Network Modules
- `transformer.py`: Contains the implementation of a transformer model used as a text encoder.
- `graph_sage.py`: Implementation of the GraphSAGE model.
- `dynamic_graph_cnn.py`: Implementation of the Dynamic Graph CNN model.
- `graph_utils.py`: Utility functions for graph operations.
- `graph_isomorphism_network.py`: Implementation of the Graph Isomorphism Network (GIN) model.
- `feature_selection_graph_neural_net.py`: Implementation of the Feature Selection Graph Neural Network (FSGNN) model.
- `graph_attention_network.py`: Implementation of the Graph Attention Network (GAT) model.
- `graph_convolutionnal_network.py`: Implementation of the Graph Convolutional Network (GCN) model.

#### `data_viz.py`
Handles data visualization using Plotly to create a 3D graph of question-graph links.

#### `train_tokenizers.py`
Script to train tokenizers for node features and prompts using the Unigram tokenizer.

#### `tokenizer.py`
Contains the implementation and training functions for the Unigram tokenizer.

#### `dataloader.py`
Defines the function to get the data loader for training and evaluation.

#### `dataset.py`
Implements a custom dataset class `CLGP_Ebiose_dataset` for handling prompt-graph pairs and preparing data for the model.

#### `data_analysis.py`
Includes functions for:
- Loading data.
- Processing data.
- Computing statistics and displaying results.

## Configuration Files

### `data_config.yaml`
Configuration for data paths and tokenizer settings:
```yaml
graph_data_file: 'Ebiose_CLGP/ebiose_clgp/data/data.jsonl'
gsm8k_validation_file: 'Ebiose_CLGP/ebiose_clgp/data/gsm8k-validation.pkl'
prompt_tokenizer: 'Ebiose_CLGP/ebiose_clgp/data_utils/prompt_tokenizer_1.json'
graph_feature_tokenizer: 'Ebiose_CLGP/ebiose_clgp/data_utils/graph_tokenizer_1.json'
prompt_context_length: 1000
node_feature_context_length: 1000
num_workers: 4
```

### `model_config.yaml`
Configuration for model settings:
```yaml
embed_dim: 256

graph_encoder:
  name: 'GCN'
  layers: 3
  hidden: 256

node_feature_encoder:
  name: 'Transformer'
  layers: 2
  heads: 2
  width: 512
  feedforward_dim: 512
  activation_function: 'gelu'
  layer_norm_eps: 1e-12
  initializer_range: 0.02

text_encoder:
  name: 'Transformer'
  layers: 2
  heads: 2
  width: 512
  feedforward_dim: 512
  activation_function: 'gelu'
  layer_norm_eps: 1e-12
  initializer_range: 0.02
```

### `train_config.yaml`
Configuration for training settings:
```yaml
per_gpu_train_batch_size: 32
logging_steps: 1
gradient_accumulation_steps: 8
num_train_epochs: 20
train_batch_size: 32
model_save_name: 'CLGP_V1.pth'
saved_checkpoints: 'Model checkpoints'
save_steps: 100

optimizer:
  lr: 0.00001
  eps: 0.00000001
  weight_decay: 0.1
```

## Usage

### Training the Model
To start training the model, run the following command:
```sh
python train.py
```
This script will:
- Load configurations from YAML files.
- Initialize the dataset and model.
- Train the model with logging, gradient clipping, and checkpointing.
- Save the trained model.


### Data Visualization
To visualize the data, run:
```sh
python data_viz.py
```

This will create a 3D graph of question-graph links using Plotly and open it in your default web browser.

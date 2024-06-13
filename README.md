# Ebiose Contrastive Language Graph Pair

## Dependencies

Before you begin, ensure your system is up-to-date and has the necessary dependencies installed. Run the following commands:

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

### `setup.py`
Defines the setup configuration for the project package.

### `train.py`
The main training script includes:
- **Loading configurations**: Sets up the environment and loads the necessary configurations.
- **Training function**: Manages the training loop, logging, and checkpointing.
- **Saving utilities**: Handles model checkpoints and other utility functions.
- **Main function**: Initializes configurations, datasets, and starts the training process.

### `utils.py`
Contains utility functions for:
- Directory creation.
- Configuration loading.
- JSON handling.

### `train_utils.py`
Includes helper functions for:
- Setting up learning rate schedules.
- Logging.
- Ensuring reproducibility by setting random seeds.

### `test_data_utils.py`
A simple script to test a tokenizer.

### `search_and_retrieval.py`
Defines functions for connecting to:
- A Neo4j graph database.
- A PostgreSQL database.
- Retrieving graph data from these databases.

### `CLGP.py`
Defines the main model class `CLGP` which includes:
- Initialization.
- Parameter settings.
- The forward pass for the model.

### `CKGGP.py`
Defines another model class `CKGGP`, a variant of the `CLGP` model.

### `transformer.py`
Contains the implementation of a transformer model used as a text encoder.

### Graph Neural Network Modules
These files contain various graph neural network models and utilities:
- `graph_sage.py`: Implementation of the GraphSAGE model.
- `dynamic_graph_cnn.py`: Implementation of the Dynamic Graph CNN model.
- `graph_utils.py`: Utility functions for graph operations.
- `graph_isomorphism_network.py`: Implementation of the Graph Isomorphism Network (GIN) model.
- `feature_selection_graph_neural_net.py`: Implementation of the Feature Selection Graph Neural Network (FSGNN) model.
- `graph_attention_network.py`: Implementation of the Graph Attention Network (GAT) model.
- `graph_convolutionnal_network.py`: Implementation of the Graph Convolutional Network (GCN) model.

### `data_viz.py`
Handles data visualization using Plotly to create a 3D graph of question-graph links.

### `train_tokenizers.py`
Script to train tokenizers for node features and prompts using the Unigram tokenizer.

### `tokenizer.py`
Contains the implementation and training functions for the Unigram tokenizer.

### `dataloader.py`
Defines the function to get the data loader for training and evaluation.

### `dataset.py`
Implements a custom dataset class `CLGP_Ebiose_dataset` for handling prompt-graph pairs and preparing data for the model.

### `data_analysis.py`
Includes functions for:
- Loading data.
- Processing data.
- Computing statistics and displaying results.

from dataset import CLGP_Ebiose_dataset
from tokenizer import train_unigram_tokenizer
import json
import pickle as pkl
import argparse

def main(graph_data_file, graph_tokenizer_name, gsm8k_validation_file, prompt_tokenizer_name):
    
    node_features_corpus = []
    prompt_corpus = []
    
    with open(graph_data_file, 'r') as f:
            graph_data = []
            for line in f:
                graph_data.append(json.loads(line))
                
    validation_set = pkl.load(open(gsm8k_validation_file, "rb"))
    prompts_data = validation_set["question"]
    
    for graph, evaluations in graph_data:
        
        graph_struct = graph['graph']
        # Extract node features
        shared_context_prompt = graph_struct.get('shared_context_prompt','')
        nodes = graph_struct.get("nodes", [])
        edges = graph_struct.get("edges", [])
        
        for node in nodes:
            node_features_corpus.append(['name:' + node.get('id', '') + '    purpose: ' + node.get('purpose', '') + '     type: ' + node.get('type','')+ '   model: ' + node.get('model','')+ '     shared_context_prompt: ' + shared_context_prompt])

        for edge in edges:
            if edge.get('condition','') != '':
                condition_node = ['name:' + edge['condition']+ '    purpose: ' + 'Allows acces to the next step if verified'+ '     type: ' + 'condition'+ '    model: ', '     shared_context_prompt: ']
                node_features_corpus.append(condition_node)
                
        for i in range(len(evaluations['evaluations'])):
                if evaluations['evaluations'][i]:  # Only consider successful evaluations
                    prompt_corpus.append(prompts_data[evaluations['dataset_indexes'][i]])
    
    graph_feature_tokenizer = train_unigram_tokenizer(node_features_corpus)
    graph_feature_tokenizer.save(f"{graph_tokenizer_name}.json")
    
    prompt_tokenizer = train_unigram_tokenizer(prompt_corpus)
    prompt_tokenizer.save(f"{prompt_tokenizer_name}.json")
    
if __name__ == '__main__':
    main('data/data.jsonl', 'data_utils/graph_tokenizer_1', 'data/gsm8k-validation.pkl', 'data_utils/prompt_tokenizer_1')

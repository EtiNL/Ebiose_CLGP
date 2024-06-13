import json
import pickle as pkl
from collections import defaultdict

def load_data(filepath):
    with open(filepath, 'r') as f:
        data = [json.loads(line) for line in f]
    return data

def load_validation_set(filepath):
    with open(filepath, 'rb') as f:
        validation_set = pkl.load(f)
    return validation_set

def process_data(data, validation_set):
    question_success_counts = defaultdict(int)
    graph_success_counts = defaultdict(int)
    question_graph_links = defaultdict(lambda: defaultdict(int))

    for entry in data:
        agent = entry[0]
        evaluations = entry[1]["evaluations"]
        dataset_indexes = entry[1]["dataset_indexes"]
        
        graph_id = agent["id"]
        for eval_result, dataset_index in zip(evaluations, dataset_indexes):
            question = validation_set["question"][dataset_index]
            if eval_result:
                question_success_counts[question] += 1
                graph_success_counts[graph_id] += 1
                question_graph_links[question][graph_id] += 1

    return question_success_counts, graph_success_counts, question_graph_links

def compute_statistics(question_success_counts, graph_success_counts, question_graph_links):
    total_questions = len(question_success_counts)
    total_graphs = len(graph_success_counts)

    most_successful_question = max(question_success_counts, key=question_success_counts.get)
    most_successful_graph = max(graph_success_counts, key=graph_success_counts.get)
    
    graph_with_most_resolutions = max(question_graph_links, key=lambda k: len(question_graph_links[k]))

    statistics = {
        "total_questions": total_questions,
        "total_graphs": total_graphs,
        "most_successful_question": {
            "question": most_successful_question,
            "success_count": question_success_counts[most_successful_question]
        },
        "most_successful_graph": {
            "graph_id": most_successful_graph,
            "success_count": graph_success_counts[most_successful_graph]
        },
        "graph_with_most_resolutions": {
            "question": graph_with_most_resolutions,
            "graph_links": question_graph_links[graph_with_most_resolutions]
        }
    }

    return statistics

def display_statistics(statistics):
    print(f"Total Questions: {statistics['total_questions']}")
    print(f"Total Graphs: {statistics['total_graphs']}")
    print("\nMost Successful Question:")
    print(f"Question: {statistics['most_successful_question']['question']}")
    print(f"Success Count: {statistics['most_successful_question']['success_count']}")
    print("\nMost Successful Graph:")
    print(f"Graph ID: {statistics['most_successful_graph']['graph_id']}")
    print(f"Success Count: {statistics['most_successful_graph']['success_count']}")
    print("\nGraph with Most Resolutions:")
    print(f"Question: {statistics['graph_with_most_resolutions']['question']}")
    # print("Graph Links:")
    # for graph_id, count in statistics['graph_with_most_resolutions']['graph_links'].items():
    #     print(f"  Graph ID: {graph_id}, Resolutions: {count}")

def main():
    data_filepath = 'ebiose_clgp/data/data.jsonl'
    validation_filepath = 'ebiose_clgp/data/gsm8k-validation.pkl'

    data = load_data(data_filepath)
    validation_set = load_validation_set(validation_filepath)

    question_success_counts, graph_success_counts, question_graph_links = process_data(data, validation_set)
    statistics = compute_statistics(question_success_counts, graph_success_counts, question_graph_links)
    display_statistics(statistics)

if __name__ == "__main__":
    main()
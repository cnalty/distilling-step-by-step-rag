import argparse
from data_utils import CQADatasetLoader, SVAMPDatasetLoader, ESNLIDatasetLoader, ANLI1DatasetLoader, ASDivDatasetLoader
from vectordb import Memory
from tqdm import tqdm

import random

def get_random_numbers(range_start, range_end, current_index, count=10):
    # Create a list of all possible numbers excluding the current index
    possible_numbers = list(range(range_start, current_index)) + list(range(current_index + 1, range_end))
    # Randomly sample 'count' numbers from the possible numbers
    return random.sample(possible_numbers, count)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--split', type=str, required=True, choices=['train', 'valid', 'test'])

args = parser.parse_args()

if args.dataset == 'cqa':
    dataset_loader = CQADatasetLoader()
elif args.dataset == 'svamp':
    dataset_loader = SVAMPDatasetLoader()
elif args.dataset == 'esnli':
    dataset_loader = ESNLIDatasetLoader()
elif args.dataset == 'anli1':
    dataset_loader = ANLI1DatasetLoader()

dataset = dataset_loader.load_from_json()[args.split]
#train_llm_rationales, train_llm_labels = dataset_loader.load_llm_preds(split='train')

if 'nli' in args.dataset:
    dataset = dataset.map(
        lambda example: {'input': example['premise'] + " " + example['hypothesis']},
        remove_columns=['premise', 'hypothesis'],
    )

memory = Memory(
    memory_file=f"./vector_db/{args.dataset}_{args.split}_rag.vecdb",
    chunking_strategy={"mode": "paragraph"},
)

with open(f'./datasets/{args.dataset}/{args.dataset}_{args.split}_matches.txt', 'w') as f:
    for i in tqdm(range(0, len(dataset['input']))):
        query = dataset['input'][i] + " " + dataset['label'][i]
        results = memory.search(query, top_n=11, unique=True)
        # matches = [result['metadata'] for result in results if result['metadata'] != i]
        matches = get_random_numbers(0, len(matches), index, count=10)
        f.write(' '.join(map(str, matches)) + "\n")

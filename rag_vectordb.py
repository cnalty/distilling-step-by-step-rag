import argparse
from data_utils import CQADatasetLoader, SVAMPDatasetLoader, ESNLIDatasetLoader, ANLI1DatasetLoader, ASDivDatasetLoader
from vectordb import Memory
from tqdm import tqdm
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--split', type=str, default='train')

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
    memory_file=f"{args.dataset}_{args.split}_rag.vecdb",
    chunking_strategy={"mode": "paragraph"},
)

batch_size = 1024
for i in tqdm(range(0, len(dataset['input']), batch_size)):
    contexts = [dataset['input'][j] + " " + dataset['label'][j] for j in range(i, min(i+batch_size, len(dataset['input'])))]
    metadata = [j for j in range(i, min(i+batch_size, len(dataset['input'])))]
    memory.save(texts=contexts, metadata=metadata)

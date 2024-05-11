import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

import argparse
from data_utils import CQADatasetLoader, SVAMPDatasetLoader, ESNLIDatasetLoader, ANLI1DatasetLoader, ASDivDatasetLoader

tokenizer = AutoTokenizer.from_pretrained('nvidia/dragon-multiturn-query-encoder')
tokenizer.add_special_tokens({'eos_token': '[EOS]'})
#query_encoder = AutoModel.from_pretrained('nvidia/dragon-multiturn-query-encoder')
context_encoder = AutoModel.from_pretrained('nvidia/dragon-multiturn-context-encoder').to('cuda')

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True)

args = parser.parse_args()

if args.dataset == 'cqa':
    dataset_loader = CQADatasetLoader()
elif args.dataset == 'svamp':
    dataset_loader = SVAMPDatasetLoader()
elif args.dataset == 'esnli':
    dataset_loader = ESNLIDatasetLoader()
elif args.dataset == 'anli1':
    dataset_loader = ANLI1DatasetLoader()

dataset = dataset_loader.load_from_json()['train']
#train_llm_rationales, train_llm_labels = dataset_loader.load_llm_preds(split='train')

if 'nli' in args.dataset:
    dataset = dataset.map(
        lambda example: {'input': example['premise'] + " " + example['hypothesis']},
        remove_columns=['premise', 'hypothesis'],
    )

batch_size = 16
with open(f'{args.dataset}_embeddings.txt', 'w') as f:
    for i in tqdm(range(0, len(dataset['input']), batch_size)):
        contexts = [dataset['input'][i] + " " + dataset['label'][i] for i in range(i, min(i+batch_size, len(dataset['input'])))]

        ctx_input = tokenizer(contexts, padding=True, truncation=True, max_length=512, return_tensors='pt').to('cuda')
        ctx_emb = context_encoder(**ctx_input).last_hidden_state[:, 0, :] # (num_ctx, emb_dim)
        for emb in ctx_emb:
            f.write(' '.join(map(str, emb.tolist())) + '\n')





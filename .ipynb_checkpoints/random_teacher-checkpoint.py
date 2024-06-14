import argparse
from data_utils import CQADatasetLoader, SVAMPDatasetLoader, ESNLIDatasetLoader, ANLI1DatasetLoader, ASDivDatasetLoader
from tqdm import tqdm
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


import random

def get_random_numbers(range_start, range_end, current_index, count=10):
    # Create a list of all possible numbers excluding the current index
    possible_numbers = list(range(range_start, current_index)) + list(range(current_index + 1, range_end))
    # Randomly sample 'count' numbers from the possible numbers
    return random.sample(possible_numbers, count)

# Example usage:


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True, choices=['cqa', 'svamp', 'anli1', 'esnli'])
parser.add_argument('--split', type=str, required=True, choices=['train', 'valid', 'test'])

args = parser.parse_args()

if args.dataset == 'cqa':
    dataset_loader = CQADatasetLoader()
elif args.dataset == 'svamp':
    dataset_loader = SVAMPDatasetLoader()
elif args.dataset == 'anli1':
    dataset_loader = ANLI1DatasetLoader()
elif args.dataset == 'esnli':
    dataset_loader = ESNLIDatasetLoader()

# Model
model_id = "nvidia/Llama3-ChatQA-1.5-8B"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype=torch.float16) # load_in_4bit=True

terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

system = "System: This is a chat between a user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions based on the context. The assistant should make use of the context to logically answer the questions"

# Dataset
dataset = dataset_loader.load_from_json()['train']


if 'nli' in args.dataset:
    dataset = dataset.map(
        lambda example: {'input': "Premise: " + example['premise'] + " Hypothesis:" + example['hypothesis']},
        remove_columns=['premise', 'hypothesis'],
    )


print("\nExecution Starts Here!")

# Reading and storing the matches
with open(f'./datasets/{args.dataset}/{args.dataset}_{args.split}_matches.txt', 'r') as f:
    matches = f.readlines()
    f.close()

final_dump = [] #list of dictionaries with question, generated label, and original label
for index in tqdm(range(len(matches))):
    match_list = get_random_numbers(0, len(matches), index, count=2)
    instance = {}
    # Generating context with matches
    final_context = ""
    for i in range(len(match_list)):
        input_val = dataset[int(match_list[i])]['input']
        label_val = dataset[int(match_list[i])]['label']
        sub_context = f"User: {input_val}\nAssistant: {label_val}\n"
        final_context += sub_context


    # Framing the prompt with context, prequestion dialogue and actual question
    if args.dataset == 'svamp':
        pre_question = "Now carefully observe how assistant logically answered above questions and answer the user question along with a rationale on how the answer is achieved. Provide me output in this form 'Rationale: ,hence the answer is Answer: '"
    elif args.dataset == 'cqa':
        pre_question = "Now carefully observe how assistant logically answered above questions and answer the user question along with a rationale on how the answer is achieved. Provide me output in this form 'Rationale: ,hence the answer is Answer: {choice}'"
    elif args.dataset == 'anli1' or args.dataset == 'esnli':
        pre_question = "Now carefully observe how assistant logically answers whether the relationship between premise and hypothesis is 'entailment', 'neutral', or 'contradiction'. The assistant also provides a rationale on how the answer is achieved. Provide me output in this form 'Rationale: ,hence the answer is Answer: {choice}'"
    else:
        raise RuntimeError

    if args.dataset == 'svamp' or args.dataset == 'cqa':
        actual_question = f"User: {dataset[index]['input']}\n\nAssistant:\n"
    elif args.dataset == 'anli1' or args.dataset == 'esnli':
        actual_question = f"User: {dataset[index]['input']}\nAnswer choices:\n(a) entailment \n(b) neutral \n(c) contradiction\n\nAssistant:\n"
    actual_label = f"\nActual_label: {dataset[index]['label']}\n"
    prompt = f"{system}\n\n{final_context}\n{pre_question}\n\n{actual_question}"

    # Inferring the Teacher with the prompt:
    tokenized_prompt = tokenizer(tokenizer.bos_token + prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(input_ids=tokenized_prompt.input_ids, 
                             attention_mask=tokenized_prompt.attention_mask, 
                             max_new_tokens=128, 
                             pad_token_id=tokenizer.eos_token_id,
                             eos_token_id=terminators)
    response = outputs[0][tokenized_prompt.input_ids.shape[-1]:]
    assistant_answer = tokenizer.decode(response, skip_special_tokens=True)

    # Preparing the dump for json
    instance['input'] = dataset[index]['input']
    instance['gen_label'] = assistant_answer
    instance['org_label'] = dataset[index]['label']
    final_dump.append(instance)
    
    # print(prompt)
    # print("instance:", instance)
    # if index == 4:
    #     break  

# Writing to json file
with open(f"./datasets/{args.dataset}/{args.dataset}_{args.split}_random_llama3.json", "w") as final:
    json.dump(final_dump, final)
    
    

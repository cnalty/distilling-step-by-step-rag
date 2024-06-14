import json
import re


split = 'test'

with open(f'svamp_{split}_llama3.json') as f:
    llama_output = json.load(f)

with open(f'svamp_{split}_matches.txt', 'r') as f:
    matches = f.readlines()

def curate(data, index):
    # Extracting fields
    gen_label = data[index]['gen_label'].lower().replace('rationale:', '').strip()
    org_label = data[index]['org_label']
    # choices = data[index]['input'].split('\n')[-5:]
    # split_choice = [re.split(r'\(.\)', choice)[1].strip() for choice in choices]
    # derived_label = "".join(choices[j] for j in range(len(split_choice)) if split_choice[j] in gen_label)

    # Splitting rationale and label
    if ('the answer is' in gen_label and len(gen_label) > 1):
        try:
            rationale, label = gen_label.split('the answer is')
        except:
            rationale = gen_label.split('the answer is')[0]
            label = ""
    else:
        rationale = gen_label
        label = ""

    # Preparing the rationale
    if rationale == "":
        rationale = gen_label.replace("the answer is","").replace('answer:', '').strip()    
    rationale = rationale.replace('hence', '').strip().strip('.')

    # Preparing the label
    if (label != "" and len(label) > 0 and bool(re.search(r'\(.\)', label))):
        label = label.replace('answer:', '').strip().strip('.')
    # elif derived_label!= "":
    #     label = derived_label
    # elif label == "":
    #     label = "".join(choices[i] for i in range(len(split_choice)) if split_choice[i] in org_label)
    
    return rationale, label


file_num = 0
dumper = []

for i in range(len(llama_output)):
    
    rationale, label = curate(llama_output, i)

    matched_num = int(matches[i].split()[0])
    rag_rationale, rag_label = curate(llama_output, matched_num)
    rag_question = llama_output[matched_num]['input']
    rag_answer = f"{rag_rationale}. So the answer is {rag_label}"
    
    entry = f"{rationale}. So the answer is {label}. Q: {rag_question} A: {rag_answer}"
    dumper.append(entry)
    
  
    if i % 1000 == 999 or i == (len(llama_output) - 1):
        with open(f"./llm/{split}_CoT_{file_num}.json", "w") as f:
            json.dump(dumper, f)
        dumper = []
        file_num += 1
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import pandas as pd
import torch
import random

template_name='prompt/LLMBar/Score.txt'
file_name='llmbar_adversarial2.csv' #data_file
df=pd.read_csv(file_name)

with open(template_name, 'r') as f:
    template = f.read()

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = "cuda" # the device to load the model onto
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")


prompts=[]
for index, row in df.iterrows():
    #direct prompting/cot
    p=template.replace('{input}', row['input']).replace('{output_1}', row['output_1']).replace('{output_2}', row['output_2'])
    
    #aspect weighting
    #p=template.replace('{input}', row['input']).replace('{auxiliary_input_0}', row['gpt4_metrics_seed200'])
    
    #aspect-wise scoring
    #p=template.replace('{input}', row['input']).replace('{output_1}', row['output_1']).replace('{output_2}', row['output_2']).replace('{auxiliary_input_0}', row['gpt4_metric3_seed200'])

    #prompted aggregation
    #p=template.replace('{input}', row['input']).replace('{output_1}', row['output_1']).replace('{output_2}', row['output_2']).replace('{q1}', row['gpt4_metric1_seed100']).replace('{q2}', row['gpt4_metric2_seed100']).replace('{q3}', row['gpt4_metric3_seed100']).replace('{q1_r}', row['gpt4_q1_seed100']).replace('{q2_r}', row['gpt4_q2_seed100']).replace('{q3_r}', row['gpt4_q3_seed100'])
    
    #aspect generation
    #p=template.replace('{input}', row['input'])
    prompts.append(p)

res=[]
for p in prompts:
    #print(p)
    #break
    messages = [
        #{"role": "user", "content": "What is your favourite condiment?"},
        #{"role": "assistant", "content": "Well, I'm quite partial to a good squeeze of fresh lemon juice. It adds just the right amount of zesty flavour to whatever I'm cooking up in the kitchen!"},
        {"role": "user", "content": p}
    ]

    encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")

    model_inputs = encodeds.to(device)
    model.to(device)

    generated_ids = model.generate(model_inputs, max_new_tokens=1000, do_sample=True, temperature=0.001)
    decoded = tokenizer.batch_decode(generated_ids)
    #print(decoded)
    #break
    print('###############', decoded[0].split('[/INST]')[1])
    res.append(decoded[0].split('[/INST]')[1].replace('</s>',''))

df['mistral_score']=res
df.to_csv(file_name, index=False)

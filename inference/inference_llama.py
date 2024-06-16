import os
import random
import time
import re
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
file_name='llmbar_adversarial2.csv' #data_file
df=pd.read_csv(file_name)
base_model_id = "meta-llama/Llama-2-13b-chat-hf"

print('Device', torch.cuda.current_device())
print([torch.cuda.device(i) for i in range(torch.cuda.device_count())])

print(base_model_id)
tokenizer = AutoTokenizer.from_pretrained(base_model_id, legcy=False, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(base_model_id, use_cache=False, torch_dtype=torch.bfloat16)
model = model.cuda().eval()

template_name='prompt/LLMBar/Score.txt'
with open(template_name, 'r') as f:
    template = f.read()
prompt_list=[]
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
    prompt_list.append(p)

#prompt_list=df['prompts'].tolist()
#prompt_list=df['p_accuracy2'].tolist()[67:68]
#prompt_list=['hi']
res=[]
with torch.no_grad():
    #while True:
        print("\n\n\n========================= Chat Session ============================")
        #chat_history = []
        #system_prompt = f"[INST] <<SYS>> You are a helpful assistant. <</SYS>> [/INST] "
        #one_prompt = system_prompt
        for p in prompt_list:
            print('prompt',p)
            chat_history = []
            system_prompt = f"[INST] <<SYS>> You are a helpful assistant in evaluating the quality of the outputs for a given instruction.  <</SYS>> [/INST] "
            one_prompt = system_prompt
            one_text = p
            chat_history.append(["USER", one_text])

            if one_text == "restart":
                break

            for one_utter in chat_history:
                if one_utter[0] == "SYS":
                    one_prompt += one_utter[1] + "</s>"
                else:
                    one_prompt += "<s>[INST] " + one_utter[1] + " [/INST] "

            input_ids = tokenizer(one_prompt, return_tensors="pt", truncation=True, add_special_tokens=False).input_ids.cuda()
            outputs = model.generate(input_ids=input_ids, min_new_tokens=2, max_new_tokens=1000, do_sample=True, top_p=0.9, temperature=0.001)
            raw_generation = tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=False)[0]
            cleaned_generation = raw_generation.split(one_text + " [/INST] ")[-1].strip()
           
            chat_history.append(["SYS", cleaned_generation])
            print("SYS >>>>>>>>> " + cleaned_generation)
            res.append(cleaned_generation)

df['llama2_score']=res
df.to_csv(file_name, index=False)

import pandas as pd 

template_name='prompt/LLMBar/Score.txt' #load the prompt template
file_name='llmbar_adversarial2.csv' #load the data file

df=pd.read_csv(file_name)


with open(template_name, 'r') as f:
    template = f.read()

#basic
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

from openai import OpenAI

client = OpenAI(
  api_key='',  # please key in your openai key
)
#prompts=df['p_score']
res=[]
for prompt in prompts:
  try:
    prompt=str(prompt)
    response = client.chat.completions.create(
                #model="gpt-3.5-turbo-0613",
                model="gpt-4-0613",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                seed=100
            )

    print(response.choices[0].message.content)
    res.append(response.choices[0].message.content.lower())
  except Exception as e:
      print(e)
      res.append(e)

df['gpt4_score_seed100']=res
df.to_csv(file_name, index=False)

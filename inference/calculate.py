import pandas as pd

file_name='llmbar_adversarial2.csv' #input your inference output file
df=pd.read_csv(file_name)

r1_scores=[]
r2_scores=[]
res=[]

for index, row in df.iterrows():

    print(index)

    w1=str(row['gpt4_w1_seed100']).replace('%','')
    w2=str(row['gpt4_w2_seed100']).replace('%','')
    w3=str(row['gpt4_w3_seed100']).replace('%','')

    e1=row['gpt4_q1_seed100'].split()
    e2=row['gpt4_q2_seed100'].split()
    e3=row['gpt4_q3_seed100'].split()
    
    #weighted sum
    r1_score=float(w1)/100*float(e1[0])+float(w2)/100*float(e2[0])+float(w3)/100*float(e3[0])
    r2_score=float(w1)/100*float(e1[1])+float(w2)/100*float(e2[1])+float(w3)/100*float(e3[1])
    
    r1_scores.append(r1_score)
    r2_scores.append(r2_score)

    #comparison
    if r1_score>r2_score:
        res.append(1)
    if r1_score<r2_score:
        res.append(2)
    if r1_score==r2_score:
        res.append(0)

df['gpt4_o1_seed100']=r1_scores
df['gpt4_o2_seed100']=r2_scores
df['gpt4_res_seed100']=res
df.to_csv('llmbar_adversarial2.csv', index=False)
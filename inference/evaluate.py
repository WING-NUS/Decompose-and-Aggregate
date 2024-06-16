import pandas as pd 

df=pd.read_csv('llmbar_adversarial_gpt4.csv')
print(df.info())

def op_to_pred(text):
    if '(a)' in text:
        return 1
    else:
        return 2

def score_to_pred(scores):
    print(scores)
    ss=scores.split()
    if len(ss)==2:
        s1, s2=float(ss[0]), float(ss[1])
    else:
        return 0
    if s1>s2:
        return 1
    elif s2>s1:
        return 2
    else:
        print('tie')
        return 0

df['pred3']=df['gpt4_score_seed100'].apply(score_to_pred)


cnt=0
for index, row in df.iterrows():
    #print(row['pred2'], row['label'])
    print(row['pred3'])
    if row['pred3']==row['label']:
        cnt+=1
print(cnt)
print(float(cnt)/df.shape[0])
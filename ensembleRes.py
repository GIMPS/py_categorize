import pandas as pd
cnt=[[0 for x in range(18)] for y in range(16111)]
for idx in range(3,12):
    vote=pd.read_csv('result'+str(idx)+'.csv')['category']
    for img in range(16111):
        cnt[img][vote[img]]+=1
f=open('ensembled_result3.csv','w')
f.write('id,category\n')

for img in range(16111):
    sel=0
    for cat in range(18):
        if cnt[img][cat]>cnt[img][sel]:
            sel=cat
    f.write(str(img+1)+','+str(sel)+'\n')
f.close
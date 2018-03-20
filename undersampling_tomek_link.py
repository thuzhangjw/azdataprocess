import pandas as pd
from imblearn.under_sampling import TomekLinks
import sys

def count(df):
    groups = df.groupby('GB/T-codename').groups
    sorted_groups = sorted(groups.items(), key=lambda x: len(x[1]), reverse=True)
    res = []
    for g in sorted_groups:
        print(g[0], len(g[1]))
        res.append((g[0], len(g[1])))
    print('\n')
    return res

def add2list(df, X, y):
    for i in range(len(df)):
        X.append(list(df.iloc[i, 2:]))
        y.append(df.iloc[i, 0])
    

def undersampling(df):
    tl = TomekLinks(ratio='all', n_jobs=16, return_indices=True)
    X = []
    y = []
    add2list(df, X, y)

    X, y, idx = tl.fit_sample(X, y)
    
    criterion = [False] * len(df)
    for i in idx:
        criterion[i] = True

    newdf = df[criterion].reset_index(drop=True)
    return newdf

df = pd.read_csv(sys.argv[1], sep='\t')
statis = count(df)

i = 1
while True: 
    print(i, 'iteration\n')
    df = undersampling(df)
    newstatis = count(df)
    if newstatis == statis:
        break
    statis = newstatis 
    i += 1
   

#newdf = pd.DataFrame.from_records(X_res, columns=list(df.columns[2:]))
#newdf = pd.concat([pd.Series(y_res, name='GB/T-codename'), newdf], axis=1)

# df.to_csv('../data/undersampled.txt', sep='\t', index=False)

## test

def undersampling_pair(df1, df2):
    tl = TomekLinks(ratio='all', n_jobs=16, return_indices=True)
    X = []
    y = []
    add2list(df1, X, y)
    add2list(df2, X, y)
    X, y, idx = tl.fit_sample(X, y)
    l1 = len(df1)
    l2 = len(df2)
    criterion1 = [False] * l1
    criterion2 = [False] * l2
    for i in idx:
        if i < l1:
            criterion1[i] = True
        else:
            criterion2[i-l1] = True
    newdf1 = df1[criterion1].reset_index(drop=True)
    newdf2 = df2[criterion2].reset_index(drop=True)
    return newdf1, newdf2 


GB_codenames = ['冠状动脉粥样硬化', '非ST段抬高型心肌梗死', '阵发性房颤', '急性前壁心肌梗死', '急性下壁心肌梗死', 
        '持续性房颤', '阵发性室上性心动过速', '冠状动脉粥样硬化性心脏病', '稳定型心绞痛']

merged_codenames = ['冠状动脉粥样硬化', '非ST段抬高型心肌梗死', '房颤', '急性心肌梗死']

dflist = []
for codename in merged_codenames:
    dflist.append(df.loc[lambda d: d['GB/T-codename'] == codename, :].reset_index(drop=True))

majoritydf = df.loc[lambda d: d['GB/T-codename'] == '不稳定型心绞痛', :].reset_index(drop=True)

statis = len(majoritydf)
while True:
    for idx, df2 in enumerate(dflist):
        majoritydf, newdf2 = undersampling_pair(majoritydf, df2)
        print(majoritydf.iloc[0, 0], len(majoritydf))
        print(newdf2.iloc[0, 0], len(newdf2))
        print('\n')
        dflist[idx] = newdf2 
    newstatis = len(majoritydf)
    if newstatis == statis:
        break
    statis = newstatis 
    
resdf = majoritydf
for df2 in dflist:
    resdf = pd.concat([resdf, df2], axis=0).reset_index(drop=True)

print('toal lines: ', len(resdf))
resdf.to_csv(sys.argv[2], sep='\t', index=False)


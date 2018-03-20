import pandas as pd
import sys

diseasemap = {
        '不稳定型心绞痛': 1,
        '冠状动脉粥样硬化': 1,
        '非ST段抬高型心肌梗死': 1,
        '阵发性房颤': '房颤',
        '持续性房颤': '房颤',
        '急性前壁心肌梗死': '急性心肌梗死',
        '急性下壁心肌梗死': '急性心肌梗死',
        '阵发性室上性心动过速': 0,
        '稳定型心绞痛': 0,
        '冠状动脉粥样硬化性心脏病': '冠状动脉粥样硬化'
        }

df = pd.read_csv(sys.argv[1], sep='\t')
criterion = [True] * len(df)

for idx, codename in enumerate(df['GB/T-codename']):
    if diseasemap[codename] == 0:
        criterion[idx] = False
    elif diseasemap[codename] != 1:
        df.loc[idx, 'GB/T-codename'] = diseasemap[codename]

df = df[criterion].reset_index(drop=True)

groups = df.groupby('GB/T-codename').groups
reslist = sorted(groups.items(), key=lambda x: len(x[1]), reverse=True)

for res in reslist:
    print(res[0], len(res[1]))

df.to_csv(sys.argv[2], sep='\t', index=False)


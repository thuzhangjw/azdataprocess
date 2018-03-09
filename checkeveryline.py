import pandas as pd

df = pd.read_excel('../data/pickedICD10.xlsx', dtype=object)
criterion = pd.notna(df['现病史'])
df = df[criterion].reset_index(drop=True)

grouped = df.groupby(lambda x: pd.notna(df.iloc[x,:]).value_counts()[True], axis = 0)
groups = grouped.groups

criterion = [True]*len(df)
for i in groups:
    print(i)
    if i >10:
        break
    for idx in groups[i]:
        criterion[idx] = False

df = df[criterion].reset_index(drop=True)

grouped = df.groupby('GB/T-codename')
groups = grouped.groups

criterion = [True] * len(df)
reslist = sorted(groups.items(), key=lambda x: len(x[1]), reverse=True)

num = 0
for res in reslist:
    num += 1
    if num > 11:
        print(res[0] + ':' + str(len(res[1])) + ' (droped)')
        for i in res[1]:
            criterion[i] = False
        continue
    print(res[0] + ':' + str(len(res[1])))

df[criterion].to_csv('../data/finalNoencoding.txt', sep='\t', index=False)


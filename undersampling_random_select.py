import pandas as pd
import random

df = pd.read_excel('../data/finalNoEncoding.xlsx')
grouped = df.groupby('GB/T-codename')
groups = grouped.groups
target = groups['不稳定型心绞痛']
criterion = [True] * len(df)

l = len(target)
num = 0
while num < 0.8 * l:
    i = random.randint(0, l-1)
    if criterion[target[i]]:
        criterion[target[i]] = False
        assert df.loc[target[i], 'GB/T-codename'] == '不稳定型心绞痛'
        num += 1

df = df[criterion].reset_index(drop=True)

grouped = df.groupby('GB/T-codename')
groups = grouped.groups
reslist = sorted(groups.items(), key=lambda x: len(x[1]), reverse=True)

for res in reslist:
    print(res[0], len(res[1]))

# generate training set
criterion = [True] * len(df)
for res in reslist:
    num = 0
    l = len(res[1])
    while num < 0.3 * l:
        i = random.randint(0, l-1)
        if criterion[res[1][i]]:
            num += 1
            criterion[res[1][i]] = False

trainingset = df[criterion].reset_index(drop=True)
testset = df[list(map(lambda x: not x, criterion))].reset_index(drop=True)

traingroups = trainingset.groupby('GB/T-codename').groups
trainlist = sorted(traingroups.items(), key=lambda x: len(x[1]), reverse=True)
for idx, val in enumerate(trainlist):
    print(val[0], len(val[1]), len(val[1])/len(reslist[idx][1]))

testgroups = testset.groupby('GB/T-codename').groups
testlist = sorted(testgroups.items(), key=lambda x: len(x[1]), reverse=True)
for idx, val in enumerate(testlist):
    print(val[0], len(val[1]), len(val[1])/len(reslist[idx][1]))

trainingset.to_csv('../data/trainingset.txt', sep='\t', index=False)
testset.to_csv('../data/testset.txt', sep='\t', index=False)

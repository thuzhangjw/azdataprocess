import pandas as pd
import random

def count(df):
    groups = df.groupby('GB/T-codename').groups
    sortedlist = sorted(groups.items(), key=lambda x: len(x[1]), reverse=True)
    for res in sortedlist:
        print(res[0], len(res[1]))


df = pd.read_csv('../data/undersampled.txt', sep='\t')

groups = df.groupby('GB/T-codename').groups
target = groups['不稳定型心绞痛']

criterion = [True] * len(df)

l = len(target)
num = 0
while num < 0.9 * l:
    i = random.randint(0, l-1)
    if criterion[target[i]]:
        criterion[target[i]] = False
        assert(df.loc[target[i], 'GB/T-codename'] == '不稳定型心绞痛')
        num += 1

df = df[criterion].reset_index(drop=True)

groups = df.groupby('GB/T-codename').groups
reslist = sorted(groups.items(), key=lambda x: len(x[1]), reverse=True)

for res in reslist:
    print(res[0], len(res[1]))
print('\n')


criterion = [True] * len(df)
for res in reslist:
    num = 0
    l = len(res[1])
    while num < 0.2 * l:
        i = random.randint(0, l-1)
        if criterion[res[1][i]]:
            num += 1
            criterion[res[1][i]] = False

train = df[criterion].reset_index(drop=True)
test = df[list(map(lambda x: not x, criterion))].reset_index(drop=True)

count(train)
print('\n')
count(test)

train.to_csv('../data/trainingset.txt', sep='\t', index=False)
test.to_csv('../data/testset.txt', sep='\t', index=False)



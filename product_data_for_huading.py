import pandas as pd
import sys

classes = ['不稳定型心绞痛', '冠状动脉粥样硬化', '房颤', '急性心肌梗死', '非ST段抬高型心肌梗死']

trainset = pd.read_csv('../data/trainingset' + sys.argv[1] + '.txt', sep='\t')
testset = pd.read_csv('../data/testset' + sys.argv[1] + '.txt', sep='\t')

wholeset = pd.concat([trainset, testset], axis=0)

numeric_data = wholeset.iloc[:, 3:].reset_index(drop=True)
disease = list(map(lambda x: classes.index(x), wholeset['GB/T-codename']))
resdf = pd.DataFrame({'disease': disease})
resdf = pd.concat([resdf, numeric_data], axis=1)

resdf.to_csv('../data/'+sys.argv[2], sep='\t', index=False)


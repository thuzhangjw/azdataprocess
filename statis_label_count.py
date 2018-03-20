import pandas as pd
import sys

df = pd.read_csv(sys.argv[1], sep ='\t')
groups = df.groupby('GB/T-codename').groups

reslist = sorted(groups.items(), key=lambda x: len(x[1]), reverse=True)

for res in reslist:
    print(res[0], len(res[1])
)

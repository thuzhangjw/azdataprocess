import pandas as pd
import sys

table = {
        '稳定性心绞痛': '稳定型心绞痛',
        '稳定心绞痛': '稳定型心绞痛',
        '抬高性心肌梗死': '抬高型心肌梗死',
        '冠状动脉动脉粥样硬化性心脏病': '冠状动脉粥样硬化性心脏病',
        '冠状动脉粥样硬化心脏病': '冠状动脉粥样硬化性心脏病'
        }


def repl(s):
    res = s
    for k in table:
        res = res.replace(k, table[k])
    return res 

log = open('../log/doc_preprocess.log', 'w')
df = pd.read_csv(sys.argv[1], sep='\t')
criterion = [True] * len(df)
num = 0
for i in range(len(df)):
    df.loc[i, '现病史'] = repl(df.loc[i, '现病史'])
    if df.loc[i, '现病史'][-1] != '。':
        criterion[i] = False
        log.write(str(i) + '    ' + df.loc[i, '现病史'] + '\n')
        num += 1

log.close()
print(num)

df[criterion].to_csv(sys.argv[2], sep='\t', index=False)


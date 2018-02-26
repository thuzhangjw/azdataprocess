import pandas as pd
import Levenshtein 
import math
from multiprocessing import Process, Manager


df = pd.read_csv('../data/pickdiagnose.txt', sep='\t', dtype=object)
diagnoseSeries = df['计算字段_出院诊断(所有)_结果_默认值']
icd10df = pd.read_excel('../data/ICD10.xlsx', dtype=object)
icd10diagnose = icd10df['名称']
newdiaglist = []

manager = Manager()
pl = []
processnum = 6
for i in range(processnum):
    pl.append(manager.list())

def processfunc(diagnoselist, reslist):
    for diagstr in diagnoselist:
        diaglist = diagstr.strip('^\t\r\n').split('^')
        newstr = ''
        for diag in diaglist:
            score = 0.0
            pos = 0
            for idx, icd in enumerate(icd10diagnose):
                ts = Levenshtein.jaro(diag.strip(), icd.strip())
                if ts > score:
                    score = ts
                    pos = idx
            newstr += diag.strip() + '(' + icd10diagnose[pos] + ':' + str(score) + ')^'
        reslist.append(newstr)

processlist = []
blocksize = math.ceil(diagnoseSeries.size/processnum)
for i in range(processnum):
    processlist.append(Process(target=processfunc, args=(diagnoseSeries[i*blocksize:(i+1)*blocksize], pl[i],)))

for i in range(processnum):
    processlist[i].start()

for i in range(processnum):
    processlist[i].join()

with open('../data/diags.txt', 'w') as f:
    for i in range(processnum):
        for s in pl[i]:
            f.write(s+'\n')


#for diagstr in diagnoseSeries:
#    diaglist = diagstr.strip('^\t\r\n').split('^')
#    newstr = ''
#    for diag in diaglist:
#        score = 0.0
#        pos = 0
#        for idx, icd in enumerate(icd10diagnose):
#            ts = Levenshtein.jaro(diag.strip(), icd.strip())
#            if ts > score:
#                score = ts
#                pos = idx
#        newstr += diag.strip() + '(' + icd10diagnose[pos] + ':' + str(score) + ')^'
#    newdiaglist.append(newstr)
#
#with open('../data/diags2.txt', 'w') as f:
#    for s in newdiaglist:
#        f.write(s+'\n')
#

import pandas as pd
import Levenshtein 
import math
from multiprocessing import Process, Manager


df = pd.read_csv('../data/pickdiagnose.txt', sep='\t', dtype=object)
diagnoseSeries = df['计算字段_出院诊断(所有)_结果_默认值']
icd10df = pd.read_excel('../data/ICD10.xlsx', dtype=object)
icd10diagnose = icd10df['名称']
newdiaglist = []

selfdict = {
    '不稳定性': '不稳定型',
    '抬高性': '抬高型',
    '房颤': '心房颤动'
}

negtivewords = ['非']

manager = Manager()
pl = []
processnum = 6
for i in range(processnum):
    pl.append(manager.list())


def processfunc(diagnoselist, reslist):
    for diagstr in diagnoselist:
        diaglist = diagstr.strip('^\t\r\n').split('^')
        newstrlist = []
        for diag in diaglist:
            newdiag = diag
            for i in selfdict:
                newdiag = newdiag.replace(i, selfdict[i])
            score = 0.0
            pos = 0
            for idx, icd in enumerate(icd10diagnose):
                ts = Levenshtein.jaro(newdiag, icd.strip())
                if ts > score:
                    score = ts
                    pos = idx
            newstrlist.append(diag + '^' + icd10diagnose[pos] + '^' + str(score))
        reslist.append(newstrlist)


processlist = []
blocksize = math.ceil(diagnoseSeries.size/processnum)
for i in range(processnum):
    processlist.append(Process(target=processfunc, args=(diagnoseSeries[i*blocksize:(i+1)*blocksize], pl[i],)))

for i in range(processnum):
    processlist[i].start()

for i in range(processnum):
    processlist[i].join()

#with open('../data/diags2.txt', 'w') as f:
#    for i in range(processnum):
#        for s in pl[i]:
#            f.write(s+'\n')

def negtivewordscheck(w1, w2):
    for nw in negtivewords:
        if (nw in w1) ^ (nw in w2):
            return False
    return True


idx = 0
diagmap = {}

def add2map(slist, i):
    if slist[1] in diagmap:
        diagmap[slist[1]][0][0] += 1
        diagmap[slist[1]][1].append((i, slist[0]))
    else:
        diagmap[slist[1]] = ([1], [(i, slist[0])])


for i in range(processnum):
    for sl in pl[i]:
        slist = sl[0].split('^')
        if float(slist[2]) > 0.8 and negtivewordscheck(slist[0], slist[1]):
            add2map(slist, idx)
#        else:
#            score = 0
#            ti = 0
#            for j in range(1, len(sl)):
#                sjlist = sl[j].split('^')
#                if float(sjlist[2]) > 0.8 and float(sjlist[2]) > score and negtivewordscheck(sjlist[0], sjlist[1]):
#                    score = float(sjlist[2])
#                    ti = j
#            if ti != 0:
#                add2map(sl[ti].split('^'), idx)
        idx += 1


reslist = sorted(diagmap.items(), key = lambda x: x[1][0][0], reverse = True)
#with open('../data/diags.txt', 'w') as f:
#    for i in range(50):
#        f.write(reslist[i][0] + '^' + str(reslist[i][1][0][0]) + '^' + str(reslist[i][1][1]) + '\n')


criterion = pd.Series([False] * len(df)) 
excludelist = ['原发性高血压']
for i in range(15):
    val = reslist[i]
    if val[0] in excludelist:
        continue
    for tp in val[1][1]:
        criterion[tp[0]] = True

df[criterion].to_csv('../data/pickedICD10.txt', sep='\t', index=False)






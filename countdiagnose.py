import pandas as pd
import Levenshtein
import math
from multiprocessing import Process, Manager


df = pd.read_csv('../data/mergedunits.txt', sep='\t', dtype=object)
diagnoseSeries = df['计算字段_出院诊断(所有)_结果_默认值']
icd10df = pd.read_excel('../data/ICD10.xlsx', dtype=object)
icd10diagnose = icd10df['名称']

selfdict = {
    '抬高性': '抬高型',
    '心房颤动': '房颤',
    '稳定性': '稳定型'
}


negtivewords = ['非']

manager = Manager()
pl = []
processnum = 6
for i in range(processnum):
    pl.append(manager.list())


def processfunc(diagnoselist, reslist):
    for diagstr in diagnoselist:
        diag = diagstr.strip('^\t\r\n').split('^')[0]
        newdiag = diag
        for w in selfdict:
            newdiag = newdiag.replace(w, selfdict[w])

        score = 0.0
        pos = 0
        for idx, icd in enumerate(icd10diagnose):
            ts = Levenshtein.ratio(newdiag, icd.strip())
            if ts > score:
                score = ts
                pos = idx

        reslist.append((diag + '^' + icd10diagnose[pos] + '^' + str(score)))

processlist = []
blocksize = math.ceil(diagnoseSeries.size / processnum)
for i in range(processnum):
    processlist.append(Process(target=processfunc, args=(diagnoseSeries[i*blocksize: (i+1)*blocksize], pl[i],)))

for i in range(processnum):
    processlist[i].start()

for i in range(processnum):
    processlist[i].join()


def negtivewordscheck(w1, w2):
    for nw in negtivewords:
       if (nw in w1) ^ (nw in w2):
           return False
       return True


diagmap = {}


def add2map(sl, i):
    if sl[1] in diagmap:
        diagmap[sl[1]][0][0] += 1
        diagmap[sl[1]][1].append((i, sl[0]))
    else:
        diagmap[sl[1]] = ([1], [(i, sl[0])])


idx = 0
for i in range(processnum):
    for s in pl[i]:
        sl = s.split('^')
        if float(sl[2]) > 0.75 and negtivewordscheck(sl[1], sl[0]):
            add2map(sl, idx)
        idx += 1

reslist = sorted(diagmap.items(), key=lambda x: x[1][0][0], reverse=True)
with open('../data/diags.txt', 'w') as f:
    for i in range(30):
        f.write(reslist[i][0] + '^' + str(reslist[i][1][0][0]) + '^' + str(reslist[i][1][1]) + '\n')


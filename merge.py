def createdict(lines, dic, idx1, idx2):
    for i in range(1, len(lines)):
        l = lines[i]
        llist = l.strip('\n').split('\t')
        keypair = (llist[idx1].strip('"'), llist[idx2].strip('"'))
#        print(str(keypair))
        dic[keypair] = l

def mergedict(emr, d1, d2, d3, d4, d):
    for key in list(d.keys()):
        if key in d1 and key in d2 and key in d3 and key in d4:
            mergedline = d1[key].strip('\n') + '\t' + d2[key].strip('\n') + '\t' + d3[key].strip('\n') + '\t' + d4[key]
            emr.append(mergedline)
            del d1[key]
            del d2[key]
            del d3[key]
            del d4[key]


f = open('../txt/EMR1_p2.txt', 'r')
lines1 = f.readlines()
f.close()

f = open('../txt/EMR2_p2.txt', 'r')
lines2 = f.readlines()
f.close()

f = open('../txt/EMR3_p2.txt', 'r')
lines3 = f.readlines()
f.close()

f = open('../txt/EMR4_p2.txt', 'r')
lines4 = f.readlines()
f.close()

#n1 = len(lines1)
#n2 = len(lines2)
#n3 = len(lines3)
#n4 = len(lines4)

target1 = u'基本信息_登记号_结果_默认值'
target2 = u'基本信息_住院号/就诊号_结果_默认值'

titlelist1 = lines1[0].strip().split('\t')
titlelist2 = lines2[0].strip().split('\t')
titlelist3 = lines3[0].strip().split('\t')
titlelist4 = lines4[0].strip().split('\t')

emr1_ti1 = titlelist1.index(target1)
emr1_ti2 = titlelist1.index(target2)

emr2_ti1 = titlelist2.index(target1)
emr2_ti2 = titlelist2.index(target2)

emr3_ti1 = titlelist3.index(target1)
emr3_ti2 = titlelist3.index(target2)

emr4_ti1 = titlelist4.index(target1)
emr4_ti2 = titlelist4.index(target2)

dic1 = {}
dic2 = {}
dic3 = {}
dic4 = {}

firstline = lines1[0].strip() + '\t' + lines2[0].strip() + '\t' + lines3[0].strip() + '\t' + lines4[0]
createdict(lines1, dic1, emr1_ti1, emr1_ti2)
createdict(lines2, dic2, emr2_ti1, emr2_ti2)
createdict(lines3, dic3, emr3_ti1, emr3_ti2)
createdict(lines4, dic4, emr4_ti1, emr4_ti2)

emr = []
mergedict(emr, dic1, dic2, dic3, dic4, dic1)
mergedict(emr, dic1, dic2, dic3, dic4, dic2)
mergedict(emr, dic1, dic2, dic3, dic4, dic3)
mergedict(emr, dic1, dic2, dic3, dic4, dic4)

#print("EMR:", len(emr))
#print("dic1:", len(dic1))
#print("dic2:", len(dic2))
#print("dic3:", len(dic3))
#print("dic4:", len(dic4))
with open('../data/mergedEMR.txt', 'w') as f:
    f.write(firstline)
    for l in emr:
        f.write(l)




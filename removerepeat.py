import sys

f = open(sys.argv[1], 'r')
lines = f.readlines()
n = len(lines)
f.close()

titlelist = lines[0].strip().split('\t')
target1 = u'基本信息_登记号_结果_默认值'
target2 = u'基本信息_住院号/就诊号_结果_默认值'
ti1 = titlelist.index(target1)
ti2 = titlelist.index(target2)

#lineslist = [0] * (n-1)
#for i in range(1, n):
#    l = lines[i].strip().split('\t')
#    keypair = (l[ti1].strip('"'), l[ti2].strip('"'))
#    lineslist[i-1] = keypair
#
#print(len(lineslist))
#print(len(set(lineslist)))


keydict = {}
for i in range(1,n):
    l = lines[i]
    ll = l.strip().split('\t')
    keypair = (ll[ti1].strip('"'), ll[ti2].strip('"'))
    if keypair in keydict:
        keydict[keypair] = 1
    else:
        keydict[keypair] = l

#num = 0
#for keypair in keydict:
#    if keydict[keypair] != 1:
#        num += 1
#print(num)
with open(sys.argv[1].split('_')[0] + '_p2.txt', 'w') as f:
    f.write(lines[0])
    for key in keydict:
        if keydict[key] != 1:
            f.write(keydict[key])


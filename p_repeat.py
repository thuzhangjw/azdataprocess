import sys

f = open(sys.argv[1], 'r')
lines = f.readlines()
f.close()

nrlines = set(lines[1:])
with open(sys.argv[1].split('.')[0] + '_p1.txt', 'w') as f:
    f.write(lines[0])
    for i in nrlines:
        f.write(i)



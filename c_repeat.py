import sys

f = open(sys.argv[1], 'r')
lines = f.readlines()
f.close()

cols = len(lines[0].strip().split('\t'))
print(cols)

for l in lines:
    n = len(l.strip().split('\t'))
    assert(n == cols)
print("Test Pass")

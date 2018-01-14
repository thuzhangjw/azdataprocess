def createdict(lines, dic, idx1, idx2):
    for l in lines:
        llist = l.strip().split('\t')
        keypair = (llist[idx1].strip('"'), llist[idx2].strip('"'))
        dic[keypair] = l

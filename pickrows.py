import numpy as np
import pandas as pd
import re
import time
import os
import psutil

numstyle = re.compile(r'^[+-]?[0-9]+(\.[0-9]*)?$')


# use Regular Expression vs try except time using
def number_filter_reg(l):
    return l.map(lambda x: False if numstyle.match(str(x)) else True)


def not_number(x):
    try:
        float(x)
        return False
    except:
        return True


def number_filter_try(l):
    return l.map(not_number) | pd.isna(l)


def get_criterion(x):
    global criterion
    #criterion &= number_filter_reg(df[x])
    criterion &= number_filter_try(df[x])


df = pd.read_excel('../data/mergedEMR.xlsx', dtype=object)
cols = df.columns.map(lambda x: x if '单位' in x else np.nan).dropna()

print(len(df))
criterion = pd.Series([True] * len(df))

t0 = time.clock()
cols.map(get_criterion)


diagnosecol = df['计算字段_出院诊断(所有)_结果_默认值']
criterion &= diagnosecol.map(lambda x: False if pd.isna(x) else True)

#print('内存使用:', psutil.Process(os.getpid()).memory_info().rss)

#s = df['检验_(URBC)尿红细胞_化验结果_全部']
#criterion2 = s.map(lambda x: not '月' in str(x) or False)
#print(criterion2.value_counts())
#
#criterion &= criterion2
#print(criterion.value_counts())
print(time.clock() - t0)
print(criterion.value_counts())
df[criterion].to_csv('../data/pickedrows.txt', sep='\t', index=False)
print('内存使用:', psutil.Process(os.getpid()).memory_info().rss)


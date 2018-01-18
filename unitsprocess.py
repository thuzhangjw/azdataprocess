from units import unitsmap  
import numpy as np
import pandas as pd
import time

def is_number(x):
    try:
        float(x)
        return True
    except:
        return False

t0 = time.clock()

df = pd.read_excel('../data/pickedrows.xlsx', dtype=object)
criterion = pd.Series([True] * len(df))
resdf = None


for col in df:
    if '化验结果' in col:
        unitname = col.replace('化验结果', '单位')
        if unitname in unitsmap:
            tmp = [0] * len(df)
            colunits = df[unitname]
            allowedunits = unitsmap[unitname]
            newcolumnname = col.replace('化验结果_全部', allowedunits[0][0])
            for idx, val in enumerate(df[col]):
                if pd.isna(val) or pd.isna(colunits[idx]):
                    tmp[idx] = np.nan
                    continue
                unitlist = str(colunits[idx]).split('^')
                valuelist = str(val).split('^')
#                if len(unitlist) != len(valuelist):
#                    print('column:', col, ' row:', idx)
#                    criterion[idx] = False
#                    tmp[idx] = np.nan
#                    continue
                    
                flag = True
                for ui, uv in enumerate(unitlist):
                    if ui >= len(valuelist):
                        break
                    if len(unitlist) == 1:
                        for vv in valuelist:
                            if uv in allowedunits[0] and is_number(vv):
                                flag = False
                                i = allowedunits[0].index(uv)
                                newdata = float(vv) / allowedunits[1][i] * allowedunits[1][0]
                                tmp[idx] = newdata
                                break
                    else:
                        if uv in allowedunits[0] and is_number(valuelist[ui]):
                            flag = False
                            i = allowedunits[0].index(uv)
                            newdata = float(valuelist[ui]) / allowedunits[1][i] * allowedunits[1][0]
                            tmp[idx] = newdata
                            break
                if flag:
                    print('column:', col, ' row:', idx)
                    tmp[idx] = np.nan
                    criterion[idx] = False
            resdf = pd.concat([resdf, pd.DataFrame({newcolumnname: tmp})], axis = 1)
        else:
            resdf = pd.concat([resdf, df[col]], axis = 1)
    elif '单位' in col:
        continue
    else:
        resdf = pd.concat([resdf, df[col]], axis = 1)

print(criterion.value_counts())
resdf.to_csv('../data/mergedunits.txt', sep='\t', index=False)
print(time.clock() - t0)


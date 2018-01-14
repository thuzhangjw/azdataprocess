import numpy as np
import pandas as pd

xlsxdf = pd.read_excel('./finalEMR.xlsx', dtype=object)
txtdf = pd.read_table('./finalEMR.txt', dtype=object)

for col in list(txtdf.columns):
    if col not in xlsxdf:
        del txtdf[col]

txtdf.to_csv('pickedEMR.txt', sep='\t')


# _*_ coding: utf-8 _*_
# @Time : 2023-04-11 14:56 
# @Author : YingHao Zhang(池塘春草梦)
# @Version：v0.1
# @File : mapping.py
# @desc :


import time

start_time = time.time()
import pandas as pd
import numpy as np
import networkx as nx

import matplotlib.pyplot as plt
import os

# %%
filepath = "D:/000大三下/BioNLP/single_paper/"
output_dir = "D:/000大三下/BioNLP/"
# filename = 'xx48478'
# filename = sys.argv[1]
# files = os.listdir(filepath)


# 去重
mapping_file = "D:/000大三下/BioNLP/result_all.txt"
mapping = pd.read_csv(mapping_file,header=None,delimiter='\t')
mapping = mapping.iloc[:,3:5]
mapping = mapping.drop_duplicates()
mapping.columns = ['entity','type']
#%%
'''
key_file = "D:/000大三下/BioNLP/result2_kw.txt"
key = pd.read_csv(key_file,header=None)
key.columns = ['entity']
key_type = pd.merge(key,mapping, on='entity')
key_type.to_csv(output_dir + 'key_type_2.txt', sep='\t', index=False, header=0)
'''
#%%
# mapping.to_csv(output_dir + 'entity_type.csv', sep='\t', index=False, header=0)
#%%
# 统计'col'列中各个数据的出现次数
counts = mapping['type'].value_counts()
print(counts)
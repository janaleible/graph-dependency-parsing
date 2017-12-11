import pandas as pd
import torch
import os
import numpy as np
from conll_df import conll_df

file_vec = ['en-ud-train.conllu', 'en-ud-dev.conllu', 'en-ud-test.conllu']

for k in range(len(file_vec)):
    path = '/home/student/Desktop/NLP/project/UD_English/' + file_vec[k]
    df = conll_df(path, file_index=False)
    df_new = df[['w', 'x', 'g', 'f']]
    data_string = df_new.to_csv()
    data_list = data_string.split('\n')
    data_list = data_list[:-1]

    names = ['training', 'dev', 'test']

    f = open("UD_english_reduced_" + names[k] + ".txt", "w")
    for i in range(len(data_list)):
        data_list[i] = data_list[i].replace('","', "mehhmeh")
        line = data_list[i].split(',')
        if (len(line) == 6):
            for j in range(len(line)):
                if (j % 6 == 2 or j % 6 == 3 or j % 6 == 4):
                    f.write(line[j] + ', ')
                if (j % 6 == 5):
                    f.write(line[j] + '\n')

    f.close()
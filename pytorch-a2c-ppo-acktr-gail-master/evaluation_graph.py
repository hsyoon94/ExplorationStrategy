import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
import scipy as sp
import json
import os

evaluation_history = np.array([])

for root, dirs, files in os.walk('/media/hsyoon/HS_VILAB1/IROS2020/200211/HalfCheetah-v2/plt0.0_plr0.0_ctrlcoef2.0/1417/log'):
    for fname in files:
        full_fname = os.path.join(root, fname)
        file_index = int(fname.split('.')[0].replace('log', ''))
        if file_index % 10 == 2:
            with open(full_fname) as json_file:
                print(json_file)
                json_data = json.load(json_file)
                print(json_data)
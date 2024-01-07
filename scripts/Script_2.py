#project-exstream
# -*- coding: utf-8 -*-
# YANN CHOHO & AGNIMO YEDIDIA

# Import librairies.
import glob
import warnings
from itertools import groupby
from typing import List, Dict, Tuple, Optional, Literal, Union, Iterable
import numpy as np
import pandas as pd
from scipy.stats import entropy
from sklearn.cluster import AgglomerativeClustering
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler


warnings.filterwarnings(action="ignore")


# Get files
DATAPATH =  "../data/custom_no_streaming_8/folder_1"
files = glob.glob(rf"{DATAPATH}/*")
files

# Read anomaly files and concatenate dataframes
dfs = []
for file in files[:-1]:
    df = pd.read_csv(file)
    # Extract the filename without the parent folder path and extension
    filename = file.split('\\')[-1].split('.')[0]
    # Add a new column "anomaly_type" with the extracted filename
    df.insert(1, column="trace_id", value=filename)
    df.rename({"Unnamed: 0": "time"}, axis=1, inplace=True)
    dfs.append(df)
    print(dfs)
# Concatenate all dataframes except labels.csv
anomaly_df = pd.concat(dfs)

# Create a separate dataframe for labels.csv
labels_df = pd.read_csv(files[-1], index_col=0)

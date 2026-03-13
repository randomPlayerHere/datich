import pandas as pd
from pathlib import Path
import os

def fileMerger(path = "data/raw"):
    files = list(Path(path).glob("*.csv"))
    dfs = []
    for file in files:
        df = pd.read_csv(file)
        dfs.append(df)
    merged_df = pd.concat(dfs, ignore_index=True)
    merged_df.sample(frac=1).reset_index(drop=True, inplace=True)
    return merged_df

def samplingDf(df, samaple_size = 5000):
    if len(df) > samaple_size:
        return df.sample(n=samaple_size, random_state=42).reset_index(drop=True)
    else:
        return df

def getData(path = "data/raw", sample_size = 5000):
    merged_df = fileMerger(path)
    sampled_df = samplingDf(merged_df, sample_size)
    return sampled_df
import pickle
from pathlib import Path
import pandas as pd


def load_pickle(filename: Path):
    return pickle.load(open(filename, 'rb'))

def dump_pickle(obj, filename):
    filename.parents[0].mkdir(parents=True, exist_ok=True)
    pickle.dump(obj, open(filename, 'wb'))

def flatten(l):
    return [item for sublist in l for item in sublist]


def to_latex(df,file):
    results = ''
    for i, row in enumerate(df.iterrows()):
        row = row[1]
        results += ' & '.join([str(v) for v in row.values]) + ((r'\\' +'\n') if i != len(df)-1 else '%\n')
    file = open(file, "w")
    a = file.write(results)
    file.close()

def concat_dataframes(dfs):
    # given a list of dataframes, concatenate them to one
    return pd.concat(dfs, ignore_index=True)
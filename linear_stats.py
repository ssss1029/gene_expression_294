import os
import numpy as np
import json
import argparse
from pathlib import Path
import pandas as pd

from numpy.core.fromnumeric import argmax


def read_json(fname, cell_id):
    f = open(fname)
    data = json.load(f)[cell_id]
    return data['test_auroc'], data['test_loss']


def read_csv(fname):
    data = pd.read_csv(fname)
    index = argmax(data['val_auroc'])
    return data['val_auroc'][index], data['val_loss'][index]


def main():

    root = args.fname
    locations = Path(root).glob('*_0.csv')
    data = {}
    ave_aur = []
    for loc in locations:
        fname = str(loc)[:-6]
        cell_id = fname[-4:]
        aur = []
        los = []
        for i in range(5):
            file = f"{fname}_{i}.csv"
            auroc, loss = read_csv(file)
            aur.append(auroc)
            los.append(loss)
        d = {
            'val_auroc': max(aur),
            'lowest_val_loss': los[argmax(aur)]
        }
        ave_aur.append(max(aur))
        data[cell_id] = d
    data['ave_auroc'] = np.mean(ave_aur)
    with open("checkpoints/linear/stats_val_max_auroc.json", "w") as outfile:
        json.dump(data, outfile, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DeepChrome")
    parser.add_argument('--fname', help='input directory', type=str)
    args = parser.parse_args()
    main()

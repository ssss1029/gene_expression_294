import os
import numpy as np
import json
import argparse
from pathlib import Path
import pandas as pd
import sys
from numpy.core.fromnumeric import argmax


def read_json(fname, cell_id):
    f = open(fname)
    data = json.load(f)[cell_id]
    return data['test_auroc'], data['test_loss']


def read_csv(fname):
    data = pd.read_csv(fname)
    index = argmax(data['val_auroc'])
    return data['val_auroc'][index], data['val_loss'][index]


def main(args):
    parser = argparse.ArgumentParser(description="DeepChrome")
    parser.add_argument('--fname', help='input directory', type=str)
    parser.add_argument('--method', help='math method', type=str)
    args = parser.parse_args(args)
    method = args.method
    root = args.fname
    locations = Path(root).glob(f'*{method}*_0.csv')
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
        idx = argmax(aur)
        file = f"{root}/test_results_{method}_{cell_id}_{idx}.json"
        test_aur, test_loss = read_json(file, fname[-4:])
        d = {
            'test_auroc': test_aur,
            'test_loss': test_loss
        }
        data[cell_id] = d
        ave_aur.append(test_aur)
    data['ave_auroc'] = np.mean(ave_aur)
    with open(f"checkpoints/linear/stats_{args.method}_val_max_auroc.json", "w") as outfile:
        json.dump(data, outfile, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    main(sys.argv[1:])
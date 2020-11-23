import os
import numpy as np
import json
import argparse
from pathlib import Path
import glob

from numpy.core.fromnumeric import argmax


def read_json(fname, cell_id):
    f = open(fname)
    data = json.load(f)[cell_id]
    return data['test_auroc'], data['test_loss']


def main():

    root = args.fname
    locations = Path(root).glob('*_0.json')
    data = {}
    for loc in locations:
        fname = str(loc)[:-7]
        cell_id = fname[-4:]
        aur = []
        los = []
        for i in range(5):
            file = f"{fname}_{i}.json"
            auroc, loss = read_json(file, cell_id)
            aur.append(auroc)
            los.append(loss)
        d = {
            'test_auroc': np.max(aur),
            'lowest_training_loss': los[argmax(aur)]
        }
        data[cell_id] = d
    with open("checkpoints/linear/stats.json", "w") as outfile:
        json.dump(data, outfile, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DeepChrome")
    parser.add_argument('--fname', help='input directory', type=str)
    args = parser.parse_args()
    main()

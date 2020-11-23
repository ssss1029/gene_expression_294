import os
import numpy as np
import json
import argparse
from pathlib import Path
import glob


def read_json(fname, cell_id):
    f = open(fname)
    data = json.load(f)[cell_id]
    return data['test_auroc']


def main():

    root = args.fname
    locations = Path(root).glob('*_0.json')
    data = {}
    for loc in locations:
        fname = str(loc)[:-7]
        cell_id = fname[-4:]
        aur = []
        for i in range(5):
            file = f"{fname}_{i}.json"
            aur.append(read_json(file, cell_id))
        d = {
            'max test_auroc': np.max(aur),
            'sd': np.std(aur)
        }
        data[cell_id] = d
    with open("checkpoints/linear/stats.json", "w") as outfile:
        json.dump(data, outfile, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DeepChrome")
    parser.add_argument('--fname', help='input directory', type=str)
    args = parser.parse_args()
    main()

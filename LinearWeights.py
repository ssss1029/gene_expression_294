import torch
from models.LinearBaseline import LinearBaseline
from pathlib import Path
import argparse
import numpy as np
import json


def load_weights(fname):
    model = LinearBaseline()
    checkpoint = torch.load(fname)
    model.load_state_dict(checkpoint['model.state_dict'])
    count = 0
    weights = 0
    bias = 0
    for param in model.parameters():
        if not count:
            weights = param.data
            count += 1
        else:
            bias = param.data
    return weights.numpy(), bias.numpy()


def main():
    root = args.fname
    locations = Path(root).glob('*_0.pth')
    data = {}
    overall_weights = []
    overall_bias = []
    for loc in locations:
        fname = str(loc)[:-6]
        cell_id = fname[-4:]
        cell_weights = []
        cell_bias = []
        for i in range(5):
            file = f"{fname}_{i}.pth"
            weights, bias = load_weights(file)
            cell_weights.append(weights)
            cell_bias.append(bias)
        mean_weights = np.mean(cell_weights, axis=0).tolist()
        mean_bias = np.mean(cell_bias, axis=0).tolist()

        data[cell_id] = {'weights': mean_weights,
                         'bias': mean_bias}
        overall_weights.append(mean_weights)
        overall_bias.append(mean_bias)
    data['ave_stats'] = {'ave weights': np.mean(overall_weights, axis=0).tolist(),
                         'ave bias': np.mean(overall_bias, axis=0).tolist()}
    with open("checkpoints/linear/model_weights.json", "w") as outfile:
        json.dump(data, outfile, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DeepChrome")
    parser.add_argument('--fname', help='input directory', type=str)
    args = parser.parse_args()
    main()

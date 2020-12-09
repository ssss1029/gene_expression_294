"""
Evaluation helpers
"""
import argparse
import os
import time

from sklearn.metrics import roc_auc_score

import torch
import torch.nn as nn
import torch.nn.functional as F
import json

from models.DeepChrome import DeepChromeModel
from dataloading.DeepChrome import DeepChromeDataset


def dict_to_gpu(d, device_id=None):
    new_dict = dict()
    for key, value in d.items():
        # Only move to GPU is cuda() is a function
        if 'cuda' in dir(value):
            new_dict[key] = value.cuda(device_id)
        else:
            new_dict[key] = value
    return new_dict


def do_evals(model, test_loader, method, no_gpu=False):
    """
    Evaluate a DeepChrome model.
    """
    model = model.eval()

    model_pos_scores = []
    groudtruth_pos_labels = []

    # Run the model through the test set
    correct = 0
    total = 0
    loss_total = 0
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            if not no_gpu:
                batch = dict_to_gpu(batch)
            batch_size = batch['X'].shape[0]

            logits = model(batch['X'], method)
            loss = F.cross_entropy(logits, batch['y'].long(), reduction='sum')
            loss_total += loss.item()

            # Update to track raw accuracy
            pred = logits.data.max(1)[1]
            correct += pred.eq(batch['y'].data).sum().item()
            total += batch_size

            # Update to track pos label scores for AUROC later.
            # Index 1 is expressed, 0 is not expressed
            pos_scores = torch.exp(logits[:, 1])
            pos_scores = pos_scores.tolist()
            model_pos_scores.extend(pos_scores)

            pos_labels = batch['y'].tolist()
            groudtruth_pos_labels.extend(pos_labels)

    AUROC = roc_auc_score(groudtruth_pos_labels, model_pos_scores)

    return AUROC, correct / total, loss_total / total


def main(args):

    assert not os.path.exists(args.log_fname)

    dset_test = DeepChromeDataset(
        dataroot=args.globstr_test,
        num_procs=args.dset_workers
    )

    test_loader = torch.utils.data.DataLoader(
        dset_test,
        batch_size=args.batch_size,
        num_workers=args.dloader_workers,
        shuffle=True,
        pin_memory=True,
    )

    model = DeepChromeModel().cuda()
    state_dict = torch.load(args.checkpoint)['model.state_dict']
    model.load_state_dict(state_dict)

    AUROC, acc, loss_avg = do_evals(model, test_loader)

    print(f"AUROC = {AUROC}. Accuracy = {acc}. Average loss = {loss_avg}")

    with open(args.log_fname, 'w') as f:
        json.dump({
            "auroc": AUROC,
            "acc": acc,
            "loss_avg": loss_avg
        }, f)

    print("Saved results")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DeepChrome Evaluation")

    # Model
    parser.add_argument('--checkpoint', default="checkpoints/TEMP/model.pth")

    # Testing data
    parser.add_argument('--globstr-test', action='append', default=[])
    parser.add_argument('--log-fname', type=str, required=True)
    # Number of workers to use to do dataloading while training.
    parser.add_argument('--dset-workers', default=24)
    # Number of workers to use to load dataset at the very beginning.
    parser.add_argument('--dloader-workers', default=24)
    parser.add_argument('--batch-size', default=1)

    main(args)

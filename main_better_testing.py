"""
Train a DeepChrome model
"""

import argparse
import os
import pprint
import time
import json
import numpy as np
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.LinearBaseline import LinearBaseline
from dataloading.DeepChrome import DeepChromeDataset

from eval import do_evals as test


def command_fname(args): return os.path.join(args.save, "command.txt")


def train_log_fname(args, cell_id, count): return os.path.join(
    args.save, f"training_log_{cell_id}_{count}.csv")


def test_results_fname(args, cell_id, count): return os.path.join(
    args.save, f"test_results_{cell_id}_{count}.json")


def checkpoint_fname(args, cell_id, count): return os.path.join(
    args.save, f"checkpoint_{cell_id}_{count}.pth")


def dict_to_gpu(d, device_id=None):
    new_dict = dict()
    for key, value in d.items():
        # Only move to GPU is cuda() is a function
        if 'cuda' in dir(value):
            new_dict[key] = value.cuda(device_id)
        else:
            new_dict[key] = value
    return new_dict


def train_one_epoch(epoch, model, dataloader, optimizer, scheduler):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    progress = ProgressMeter(
        len(dataloader),
        [batch_time, data_time, losses],
        prefix="Epoch: [{}]".format(epoch)
    )

    loss_moving_average = 0
    model.train()

    end = time.time()
    for i, batch in enumerate(dataloader):
        data_time.update(time.time() - end)

        if not args.no_gpu:
            batch = dict_to_gpu(batch)
        batch_size = batch['X'].shape[0]

        # import pdb; pdb.set_trace()

        optimizer.zero_grad()

        logits = model(batch['X'])
        loss = F.cross_entropy(logits, batch['y'].long())
        loss.backward()
        optimizer.step()
        scheduler.step()

        losses.update(loss.item(), batch_size)
        loss_moving_average = (0.1 * loss.item()) + (0.9 * loss_moving_average)

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0 and i > 0:
            progress.display(i)

    return losses.avg


def main():
    # Dataloading

    dset_train = DeepChromeDataset(
        dataroot=args.globstr_train,
        num_procs=args.dset_workers
    )
    print(f"Training set has {len(dset_train)} samples.")

    train_loader = torch.utils.data.DataLoader(
        dset_train,
        batch_size=args.batch_size,
        num_workers=args.dloader_workers,
        shuffle=True,
        pin_memory=True,
    )

    for count in range(5):
        print(pprint.pformat(vars(args)))

        # Bookkeeping
        # if os.path.exists(test_results_fname(args, count)):
        #     resp = None
        #     while resp not in {"yes", "no", "y", "n"}:
        #         resp = input(f"{args.save} already exists. Overwrite contents? [y/n]: ")
        #         if resp == "yes" or resp == "y":
        #             break
        #         elif resp == "no" or resp =="n":
        #             print("Exiting")
        #             exit()
        # else:
        os.makedirs(args.save, exist_ok=True)

        # Save command to file
        with open(command_fname(args), 'w') as f:
            f.write(pprint.pformat(vars(args)))

        # Setup Model
        model = LinearBaseline()

        if not args.no_gpu:
            model = model.cuda()

        # Optimization
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.wd,
            momentum=args.momentum
        )

        def cosine_annealing(step, total_steps, lr_max, lr_min):
            return lr_min + (lr_max - lr_min) * 0.5 * (
                1 + np.cos(step / total_steps * np.pi))

        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: cosine_annealing(
                step,
                args.epochs * len(train_loader),
                1,  # since lr_lambda computes multiplicative factor
                1e-6 / args.lr
            )
        )

        # Logging
        with open(train_log_fname(args, args.globstr_val_cell_ids[0], count), 'w') as f:
            f.write("epoch,train_loss,val_loss,val_acc,val_auroc\n")

        # Train!
        print("Beginning training...")
        best_epoch_auroc = 0
        best_epoch = None
        num_without_changing_best_val_auroc = 0

        TestCells = []
        for cell in args.globstr_val_cell_ids:
            TestCells.append(CellDataSet(
                cell,
                checkpoint_fname(args, cell, count),
                args.batch_size,
                args.dset_workers,
                args.dloader_workers
            ))

        for epoch in range(args.epochs):
            if all(map(lambda cell: cell.is_done, TestCells)):
                break

            # Train 1 epoch
            train_loss = train_one_epoch(
                epoch, model, train_loader, optimizer, scheduler)
            print("1")
            # Validate
            # val_auroc, val_acc, val_loss = test(model, val_loader, args.no_gpu)
            total_val_auroc = 0
            total_val_acc = 0
            total_val_loss = 0
            num_cells = 0
            for cell in TestCells:
                if cell.is_done:
                    continue

                num_cells += 1
                print('2')
                val_auroc, val_acc, val_loss = test(
                    model, cell.val_loader, args.no_gpu)

                total_val_auroc += val_auroc
                total_val_acc += val_acc
                total_val_loss += val_loss

                cell.add_valid_auroc(val_auroc, epoch, model.state_dict(
                ), optimizer.state_dict(), args.patience)
            print('3')
            # Logging
            # print('Epoch {0:3d} | Train Loss {1:.6f} | Val Loss {2:.6f} | Val AUROC {3:.6f} | Val Accuracy {4:.6f}'.format(
            #     epoch,
            #     train_loss,
            #     total_val_auroc / num_cells,
            #     total_val_acc / num_cells,
            #     total_val_loss / num_cells,
            # ))

            with open(train_log_fname(args, args.globstr_val_cell_ids[0], count), 'a') as f:
                f.write(
                    f"{epoch},{train_loss},{total_val_loss / num_cells},{total_val_acc / num_cells},{total_val_auroc / num_cells}\n")
            print('4')
        # Save the stragglers
        for cell in TestCells:
            if not cell.is_done:
                print(f"{cell.cell_id} was a straggler :(")
                cell._save_model_to_disk()

        # Test on all cells
        all_save_data = dict()
        for cell in TestCells:
            model.load_state_dict(cell.best_model)

            print(f"Doing final testing on cell {cell.cell_id}")

            # Do final testing
            test_auroc, test_acc, test_loss = test(
                model, cell.test_loader, args.no_gpu)
            data = {
                "test_auroc": test_auroc,
                "test_acc": test_acc,
                "test_loss": test_loss
            }
            all_save_data[cell.cell_id] = data

        with open(test_results_fname(args, args.globstr_val_cell_ids[0], count), 'w', encoding='utf-8') as f:
            print(pprint.pformat(all_save_data))
            json.dump(all_save_data, f, ensure_ascii=False, indent=4)

    print(f"Finished successfully. See {args.save}")


##################################################################
# Support Classes
##################################################################

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


class CellDataSet:
    def __init__(self, cell_id, checkpoint_path, batch_size, dset_workers, dloader_workers):
        self.cell_id = cell_id
        self.dset_workers = dset_workers
        self.dloader_workers = dloader_workers
        self.batch_size = batch_size
        self.checkpoint_path = checkpoint_path
        self.val_loader, self.test_loader = self.__prepare_data_loaders()
        self.is_done = False
        self.best_auroc = -float('inf')
        self.best_epoch = -1
        self.best_model = {}
        self.best_opt = {}
        self.since_last_update = 0

    def add_valid_auroc(self, auroc, epoch, model_state_dict, opt_state_dict, patience):
        assert not self.is_done, "This cell is done training."
        if auroc > self.best_auroc:
            self.since_last_update = 0
            self.best_auroc = auroc
            self.best_epoch = epoch
            self.best_model = copy.deepcopy(model_state_dict)
            self.best_opt = copy.deepcopy(opt_state_dict)
        else:
            self.since_last_update += 1

        if self.since_last_update >= patience:
            self.is_done = True
            self._save_model_to_disk()

    def _save_model_to_disk(self):
        _dict = {
            "model.state_dict": self.best_model,
            "optimizer.state_dict": self.best_opt,
            "epoch": self.best_epoch,
        }
        os.makedirs(os.path.dirname(self.checkpoint_path), exist_ok=True)
        torch.save(_dict, self.checkpoint_path)
        print("Saved model for {} to disk!".format(self.cell_id))

    def __prepare_data_loaders(self):
        dset_val = DeepChromeDataset(
            dataroot=[
                './dataset/{}/classification/valid.csv'.format(self.cell_id)],
            num_procs=self.dset_workers
        )
        print(f"Validation set has {len(dset_val)} samples.")

        dset_test = DeepChromeDataset(
            dataroot=[
                './dataset/{}/classification/test.csv'.format(self.cell_id)],
            num_procs=self.dset_workers
        )
        print(f"Test set has {len(dset_test)} samples.")

        val_loader = torch.utils.data.DataLoader(
            dset_val,
            batch_size=self.batch_size,
            num_workers=self.dloader_workers,
            shuffle=True,
            pin_memory=True,
        )

        test_loader = torch.utils.data.DataLoader(
            dset_test,
            batch_size=self.batch_size,
            num_workers=self.dloader_workers,
            shuffle=True,
            pin_memory=True,
        )

        return val_loader, test_loader


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DeepChrome")
    parser.add_argument('--threads', default=2, type=int)
    parser.add_argument('--epochs', default=100, type=int)

    # Data
    parser.add_argument('--globstr-train', action='append', default=[])
    parser.add_argument('--globstr-val-cell-ids', action='append', default=[])
    # Number of workers to use to load dataset at the very beginning.
    parser.add_argument('--dset-workers', default=1)
    # Number of workers to use to do dataloading while training.
    parser.add_argument('--dloader-workers', default=0)
    parser.add_argument('--trsize', default="10")
    parser.add_argument('--tssize', default="10")

    # Model
    parser.add_argument('--arch', default='DeepChrome',
                        choices=['DeepChrome', 'DeepChromeFC'])
    parser.add_argument('--nonlinearity', default='relu')
    parser.add_argument('--nhus', default=128, type=int)
    parser.add_argument('--ipdim', default=1, type=int)
    parser.add_argument('--opdim1', default=6, type=int)
    parser.add_argument('--opdim2', default=16, type=int)
    parser.add_argument('--FCnhus1', default=120, type=int)
    parser.add_argument('--FCnhus2', default=84, type=int)
    parser.add_argument('--conv_kernels', default=5, type=int)
    parser.add_argument('--pools', default=2, type=int)

    # Training
    parser.add_argument('--batch-size', default=32, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--wd', default=0, type=float)
    parser.add_argument('--momentum', default=0, type=float)
    parser.add_argument('--no-gpu', default=True, action='store_true')
    # Early stopping patience
    parser.add_argument('--patience', default=15, type=int)

    # Logging
    parser.add_argument('--print-freq', default=50, type=int)
    parser.add_argument('--save', default="checkpoints/linear")

    args = parser.parse_args()

    main()

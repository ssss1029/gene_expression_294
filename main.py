"""
Train a DeepChrome model
"""

import argparse
import os
import pprint
import time
import json

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.DeepChrome import DeepChromeModel
from models.DeepChromeFC import DeepChromeFCModel
from dataloading.DeepChrome import DeepChromeDataset

from eval import do_evals as test

command_fname       = lambda args: os.path.join(args.save, "command.txt")
train_log_fname     = lambda args: os.path.join(args.save, "training_log.csv")
test_results_fname  = lambda args: os.path.join(args.save, "test_results.json")
checkpoint_fname    = lambda args, epoch: os.path.join(args.save, "checkpoint.pth")

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

    print(pprint.pformat(vars(args)))

    ###### Bookkeeping
    if os.path.exists(args.save):
        resp = None
        while resp not in {"yes", "no", "y", "n"}:
            resp = input(f"{args.save} already exists. Overwrite contents? [y/n]: ")
            if resp == "yes" or resp == "y":
                break
            elif resp == "no" or resp =="n":
                print("Exiting")
                exit()
    else:
        os.makedirs(args.save, exist_ok=True)
    
    # Save command to file
    with open(command_fname(args), 'w') as f:
        f.write(pprint.pformat(vars(args)))


    ###### Dataloading

    dset_train = DeepChromeDataset(
        dataroot=args.globstr_train,
        num_procs=args.dset_workers
    )
    print(f"Training set has {len(dset_train)} samples.")

    dset_val = DeepChromeDataset(
        dataroot=args.globstr_val,
        num_procs=args.dset_workers
    )
    print(f"Validation set has {len(dset_val)} samples.")

    dset_test = DeepChromeDataset(
        dataroot=args.globstr_test,
        num_procs=args.dset_workers
    )
    print(f"Test set has {len(dset_test)} samples.")

    train_loader = torch.utils.data.DataLoader(
        dset_train, 
        batch_size=args.batch_size, 
        num_workers=args.dloader_workers, 
        shuffle=True,
        pin_memory=True,
    )

    val_loader = torch.utils.data.DataLoader(
        dset_val, 
        batch_size=args.batch_size, 
        num_workers=args.dloader_workers, 
        shuffle=True,
        pin_memory=True,
    )

    test_loader = torch.utils.data.DataLoader(
        dset_test, 
        batch_size=args.batch_size, 
        num_workers=args.dloader_workers, 
        shuffle=True,
        pin_memory=True,
    )

    ###### Setup Model
    if args.arch == 'DeepChrome':
        model = DeepChromeModel()
    elif args.arch == 'DeepChromeFC':
        model = DeepChromeFCModel()
    else:
        raise NotImplementedError()

    if not args.no_gpu:
        model = model.cuda()

    ###### Optimization
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


    ###### Logging
    with open(train_log_fname(args), 'w') as f:
        f.write("epoch,train_loss,val_loss,val_acc,val_auroc\n")

    ###### Train!
    print("Beginning training...")
    best_epoch_auroc = 0
    best_epoch = None
    num_without_changing_best_val_auroc = 0
    for epoch in range(args.epochs):
        
        train_loss = train_one_epoch(epoch, model, train_loader, optimizer, scheduler)

        val_auroc, val_acc, val_loss = test(model, val_loader, args.no_gpu)

        ###### Logging

        print('Epoch {0:3d} | Train Loss {1:.6f} | Val Loss {2:.6f} | Val AUROC {3:.6f} | Val Accuracy {4:.6f}'.format(
            (epoch + 1),
            train_loss,
            val_loss,
            val_auroc,
            val_acc,
        ))

        with open(train_log_fname(args), 'a') as f:
            f.write(f"{epoch},{train_loss},{val_loss},{val_acc},{val_auroc}\n")

        if val_auroc > best_epoch_auroc:
            best_epoch = epoch
            best_epoch_auroc = val_auroc
            num_without_changing_best_val_auroc = 0
            # Save the model iff this is the best epoch so far
            _dict = {
                "model.state_dict" : model.state_dict(),
                "optimizer.state_dict" : optimizer.state_dict(),
                "epoch" : epoch,
            }
            torch.save(_dict, checkpoint_fname(args, epoch))
        else:
            num_without_changing_best_val_auroc += 1
        
        if num_without_changing_best_val_auroc > args.patience:
            print("Early stopping")
            break

    print(f"Doing final testing")

    print("Loading {0}".format(checkpoint_fname(args, best_epoch)))
    model.load_state_dict(torch.load(checkpoint_fname(args, best_epoch))['model.state_dict'])

    # Do final testing
    test_auroc, test_acc, test_loss = test(model, test_loader, args.no_gpu)
    with open(test_results_fname(args), 'w', encoding='utf-8') as f:
        data = {
            "test_auroc" : test_auroc, 
            "test_acc" : test_acc, 
            "test_loss" : test_loss
        }
        print(pprint.pformat(data))
        json.dump(data, f, ensure_ascii=False, indent=4)


    print(f"Finished successfully. See {args.save}")


##################################################################
###### Support Classes
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DeepChrome")
    parser.add_argument('--threads', default=2)
    parser.add_argument('--epochs', default=100)
      
    # Data
    parser.add_argument('--globstr-train', action='append', default=[])
    parser.add_argument('--globstr-val', action='append', default=[])
    parser.add_argument('--globstr-test', action='append', default=[])
    parser.add_argument('--dset-workers', default=24) # Number of workers to use to load dataset at the very beginning.
    parser.add_argument('--dloader-workers', default=10) # Number of workers to use to do dataloading while training.
    parser.add_argument('--trsize', default="10")
    parser.add_argument('--tssize', default="10")
    
    # Model
    parser.add_argument('--arch', default='DeepChrome', choices=['DeepChrome', 'DeepChromeFC'])
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
    parser.add_argument('--batch-size', default=256, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--wd', default=0, type=float)
    parser.add_argument('--momentum', default=0, type=float)
    parser.add_argument('--no-gpu', action='store_true')
    parser.add_argument('--patience', default=4, type=int) # Early stopping patience

    # Logging
    parser.add_argument('--print-freq', default=50, type=int)
    parser.add_argument('--save', default="checkpoints/TEMP")

    args = parser.parse_args()

    main()
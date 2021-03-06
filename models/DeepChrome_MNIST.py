
"""
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepChromeMNISTModel(torch.nn.Module):
    def __init__(self):
        """
        DeepChrome model.
        Comments are shape descriptions. 
        """
        super(DeepChromeMNISTModel, self).__init__()
        
        # TODO: If we want to change these, add command line args.
        kernel_size = 5 # used to be 5
        num_filters = 50
        pool_size = 5 # used to be 5
        mlp_h1 = 728 # used to be 625
        mlp_h2 = 125 # used to be 125
        noutputs = 10

        self.stage1 = nn.Sequential(
            nn.Conv2d(1, num_filters, kernel_size=(kernel_size, kernel_size)),
            nn.ReLU(),
        )

        self.stage2 = nn.Sequential(
            nn.MaxPool2d(pool_size)
        )

        self.stage3 = nn.Sequential(
            nn.Dropout(0.5), # When changing this, check if this is the prob of keeping something or the prob of dropping it.
            nn.Linear(800, mlp_h1),
            nn.ReLU(),
            nn.Linear(mlp_h1, mlp_h2),
            nn.ReLU(),
            nn.Linear(mlp_h2, noutputs)
        )
    
    def forward(self, x):
        """
        Do a forward pass
        """
        batch_size = x.shape[0]

        print(x.shape)

        x = self.stage1(x) 
        x = x.squeeze(3) 

        print(x.shape)

        x = self.stage2(x)
        x = x.reshape((batch_size, -1))

        print(x.shape)

        x = self.stage3(x)
        
        return x

if __name__ == "__main__":
    # Do some sanity check
    model = DeepChromeMNISTModel().train()
    print(f"NUM PARAMETERS = {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    X = torch.ones((3, 1, 28, 28)) # Example batch size of 3 
    retval = model(X)
    print(retval)
    print(retval.shape)
    import pdb; pdb.set_trace()

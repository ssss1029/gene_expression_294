
"""
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepChromeFCModel(torch.nn.Module):
    def __init__(self):
        """
        DeepChrome model.
        Comments are shape descriptions. 
        """
        super(DeepChromeFCModel, self).__init__()
        
        # TODO: If we want to change these, add command line args.
        mlp_h0 = 2048
        mlp_h1 = 1024
        mlp_h2 = 625
        mlp_h3 = 125
        noutputs = 2

        # [B, 1, 100, 5]
        self.stage1 = nn.Sequential(
            nn.Linear(100 * 5, mlp_h0),
            nn.ReLU(),
        )

        self.stage3 = nn.Sequential(
            nn.Linear(mlp_h0, mlp_h1),
            nn.Dropout(0.5), # When changing this, check if this is the prob of keeping something or the prob of dropping it.
            nn.Linear(mlp_h1, mlp_h2),
            nn.ReLU(),
            nn.Linear(mlp_h2, mlp_h3),
            nn.ReLU(),
            nn.Linear(mlp_h3, noutputs)
        )
    
    def forward(self, x):
        """
        Do a forward pass
        X is given as [B, 100, 5]
        """
        batch_size = x.shape[0]
        x = x.reshape((batch_size, -1))
        x = self.stage1(x)
        x = self.stage3(x)
        return x

if __name__ == "__main__":
    # Do some sanity check
    model = DeepChromeFCModel().train()
    print(f"NUM PARAMETERS = {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    X = torch.ones((3, 100, 5)) # Example batch size of 3 
    retval = model(X)
    print(retval)
    import pdb; pdb.set_trace()

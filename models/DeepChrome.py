
"""
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepChromeModel(torch.nn.Module):
    def __init__(self):
        """
        DeepChrome model.
        Comments are shape descriptions. 
        """
        super(DeepChromeModel, self).__init__()
        
        # TODO: If we want to change these, add command line args.
        kernel_size = 10
        num_filters = 50
        pool_size = 5
        mlp_h1 = 625
        mlp_h2 = 125
        noutputs = 2

        # [B, 1, 100, 5]
        self.stage1 = nn.Sequential(
            nn.Conv2d(1, num_filters, kernel_size=(kernel_size, 5)),
            nn.ReLU(),
        )
        # [B, num_filters, 100 - kernel_size, 1]

        # [B, num_filters, 100 - kernel_size]
        self.stage2 = nn.Sequential(
            nn.AvgPool1d(pool_size)
        )
        # [B, num_filters, math.floor((100 - kernel_size - pool_size) / pool_size + 1)]

        # [B, num_filters * math.floor((100 - kernel_size - pool_size) / pool_size + 1)]
        self.stage3 = nn.Sequential(
            nn.Dropout(0.5), # When changing this, check if this is the prob of keeping something or the prob of dropping it.
            nn.Linear(num_filters * math.floor((100 - kernel_size - pool_size) / pool_size + 1), mlp_h1),
            nn.ReLU(),
            nn.Linear(mlp_h1, mlp_h2),
            nn.ReLU(),
            nn.Linear(mlp_h2, noutputs)
        )
        # [B, noutputs]
    
    def forward(self, x):
        """
        Do a forward pass
        X is given as [B, 100, 5]
        """
        batch_size = x.shape[0]

        x = x.unsqueeze(1) # [B, 1, 100, 5]
        assert x.shape[1] == 1

        x = self.stage1(x) # [B, num_filters, 100 - kernel_size, 1]
        x = x.squeeze(3) # [B, num_filters, 100 - kernel_size]

        x = self.stage2(x)
        x = x.reshape((batch_size, -1))
        
        x = self.stage3(x)
        
        return x

if __name__ == "__main__":
    # Do some sanity check
    model = DeepChromeModel().train()
    print(f"NUM PARAMETERS = {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    X = torch.ones((3, 100, 5)) # Example batch size of 3 
    retval = model(X)
    print(retval)
    import pdb; pdb.set_trace()

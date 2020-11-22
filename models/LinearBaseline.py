import torch
import torch.nn as nn
from torchsummary import summary

class LinearBaseline(nn.Module):
    def __init__(self):
        """
        Linear baseline model.
        """
        super(LinearBaseline, self).__init__()

        self.layer = nn.Linear(5, 2)

    def average(self, x):
        ans = torch.mean(x, dim=1)
        return ans

    def forward(self, x):
        x = self.average(x)
        x = self.layer(x)
        return x


if __name__ == "__main__":
    model = LinearBaseline().train()
    print(
        f"NUM PARAMETERS = {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    X = torch.ones((25, 100, 5))  # Example batch size of 3
    retval = model(X)
    print(summary(model, X, device='cpu'))

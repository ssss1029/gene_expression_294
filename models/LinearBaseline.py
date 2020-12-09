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

    def action(self, x, func, method):
        ans = func(x, dim=1)
        if len(ans) == 2:
            ans, _ = ans
        if method == 'sum' or method == 'max':
            ans /= torch.max(ans)
        return ans

    def forward(self, x, method):
        act = 0
        if method == 'median':
            act = torch.median
        elif method == 'min':
            act = torch.min
        elif method == 'max':
            act = torch.max
        elif method == 'sum':
            act = torch.sum
        x = self.action(x, act, method)
        x = self.layer(x)
        return x


if __name__ == "__main__":
    model = LinearBaseline().train()
    print(
        f"NUM PARAMETERS = {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    X = torch.ones((25, 100, 5))  # Example batch size of 3
    retval = model(X)
    print(summary(model, X, device='cpu'))

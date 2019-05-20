import torch
import torch.nn as nn

from . import loss_utils


def test_moving_stats():
    """Optimizing KLD on Moving Stats leads normal distribution"""

    model = nn.Linear(1, 1)
    ms = loss_utils.MovingStats(momentum=0.9)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # Initially, model maps N(0, 1) to N(10, 3), which should become
    # close to N(0, 1) as KLD is optimized.
    model.weight.data.fill_(3)
    model.bias.data.fill_(10)
    # Hack for testing
    ms.mean = 10
    ms.var = 3

    for i in range(10000):
        x = torch.randn(32, 1, requires_grad=True)
        y = model(x)
        mean, var = ms(y, update=True)
        kld = torch.mean(loss_utils.kld_loss(mean, var))
        model.zero_grad()
        kld.backward()
        optimizer.step()

        if i % 500 == 0:
            print(mean.item(), var.item(), kld.item())

    assert abs(mean.item()) < 0.3
    assert abs(var.item() - 1) < 0.3
    assert kld.item() < 0.3

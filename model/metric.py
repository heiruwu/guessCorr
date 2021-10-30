import torch


def acc(output, target):
    # acc by thresh 0.1
    with torch.no_grad():
        assert output.shape[0] == len(target)
        correct = torch.sum(abs(target-output) <= 0.1).item()
    return correct / len(target)


def mape(output, target):
    # mean absolute percentage error
    with torch.no_grad():
        assert output.shape[0] == len(target)
        mape = torch.sum(abs((target-output)/target)).item()
    return mape / len(target)

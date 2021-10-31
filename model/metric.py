import torch


def MSELoss(yhat, y):
    return torch.nn.functional.mse_loss(yhat, y)

def acc(output, target):
    # acc by thresh 0.1
    with torch.no_grad():
        assert output.shape[0] == len(target)
        correct = torch.sum(abs(target-output) <= 0.05).item()
    return correct / len(target)

def mse(output, target):
    # mean square error
    with torch.no_grad():
        assert output.shape[0] == len(target)
        mae = torch.sum(MSELoss(output, target)).item()
    return mae / len(target)

def mae(output, target):
    # mean absolute error
    with torch.no_grad():
        assert output.shape[0] == len(target)
        mae = torch.sum(abs(target-output)).item()
    return mae / len(target)

def mape(output, target):
    # mean absolute percentage error
    with torch.no_grad():
        assert output.shape[0] == len(target)
        mape = torch.sum(abs((target-output)/target)).item()
    return mape / len(target)

def rmse(output, target):
    # mean absolute percentage error
    with torch.no_grad():
        assert output.shape[0] == len(target)
        rmse = torch.sum(torch.sqrt(MSELoss(output, target))).item()
    return rmse / len(target)

def msle(output, target):
    # mean absolute percentage error
    with torch.no_grad():
        assert output.shape[0] == len(target)
        msle = torch.sum(MSELoss(torch.log(output + 1), torch.log(target + 1))).item()
    return msle / len(target)

def rmsle(output, target):
    # mean absolute percentage error
    with torch.no_grad():
        assert output.shape[0] == len(target)
        rmsle = torch.sum(torch.sqrt(MSELoss(torch.log(output + 1), torch.log(target + 1)))).item()
    return rmsle / len(target)

import torch


def SmoothL1(yhat, y):
    return torch.nn.functional.smooth_l1_loss(yhat, y)

def MSELoss(yhat, y):
    return torch.nn.functional.mse_loss(yhat, y)

def RMSELoss(yhat, y):
    return torch.sqrt(MSELoss(yhat, y))

def MSLELoss(yhat, y):
    return MSELoss(torch.log(yhat + 1), torch.log(y + 1))

def RMSLELoss(yhat, y):
    return torch.sqrt(MSELoss(torch.log(yhat + 1), torch.log(y + 1)))

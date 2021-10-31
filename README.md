# Piccollage take home quiz

## guess the correlation
### data inspection
a pretty normal distribution

![dist](https://github.com/gt212345/piccollage/blob/main/materials/dist.png?raw=true)

### train/val/test split
splitting amount
```bash
.dataset:                150000 instances
├─80%─├─80%─training      96000 instances
│     └─20%─validation    24000 instances
├─20%─testing             30000 instances

```
after a rough glance at the dataset distribution, I considered the dataset is pretty normal distributed and has enough instances to keep the variance low after 80/20 splitting.

splitting method
```python
def _split_dataset(self, split, training=True):
    if split == 0.0:
        return None, None

    # self.correlations_frame = pd.read_csv('path/to/csv_file')
    n_samples = len(self.correlations_frame)

    idx_full = np.arange(n_samples)

    # fix seed for referenceable testing set
    np.random.seed(0)
    np.random.shuffle(idx_full)

    if isinstance(split, int):
        assert split > 0
        assert split < n_samples, "testing set size is configured to be larger than entire dataset."
        len_test = split
    else:
        len_test = int(n_samples * split)

    test_idx = idx_full[0:len_test]
    train_idx = np.delete(idx_full, np.arange(0, len_test))

    if training:
        dataset = self.correlations_frame.ix[train_idx]
    else:
        dataset = self.correlations_frame.ix[test_idx]

    return dataset
```
training/validation splitting uses the same logic
### model inspection
```bash
CorrelationModel(
  (features): Sequential(
    (0): Conv2d(1, 16, kernel_size=(3, 3), stride=(2, 2), padding=(2, 2))
    #(0): params: (3*3*1+1) * 16 = 160
    (1): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #(1): params: 16 * 2 = 32
    (2): ReLU(inplace=True)
    (3): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
    (4): Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2))
    #(4): params: (3*3*16+1) * 32 = 4640
    (5): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #(5): params: 32 * 2 = 64
    (6): ReLU(inplace=True)
    (7): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
    (8): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    #(8): params: (3*3*32+1) * 64 = 18496
    (9): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #(9): params: 64 * 2 = 128
    (10): ReLU(inplace=True)
    (11): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
    (12): Conv2d(64, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    #(12): params: (3*3*64+1) * 32 = 18464
    (13): ReLU(inplace=True)
    (14): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
    (15): Conv2d(32, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (#15): params: (3*3*32+1) * 16 = 4624
    (16): ReLU(inplace=True)
    (17): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
    (18): Conv2d(16, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (#18): params: (3*3*16+1) * 8 = 1160
    (19): ReLU(inplace=True)
  )
  (avgpool): AdaptiveAvgPool2d(output_size=(1, 1))
  (linear): Sequential(
    (0): Conv2d(8, 1, kernel_size=(1, 1), stride=(1, 1))
    #(0): params: (8+1) * 1 = 9
    (1): Tanh()
  )
)
Trainable parameters: 47777
```
### loss function
```python
def MSELoss(yhat, y):                                                  <--- final choice
    return torch.nn.functional.mse_loss(yhat, y)

def RMSELoss(yhat, y):
    return torch.sqrt(MSELoss(yhat, y))

def MSLELoss(yhat, y):
    return MSELoss(torch.log(yhat + 1), torch.log(y + 1))

def RMSLELoss(yhat, y):
    return torch.sqrt(MSELoss(torch.log(yhat + 1), torch.log(y + 1)))

```
### evaluation metric
```python
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
    # root mean square error
    with torch.no_grad():
        assert output.shape[0] == len(target)
        rmse = torch.sum(torch.sqrt(MSELoss(output, target))).item()
    return rmse / len(target)

def msle(output, target):
    # mean square log error
    with torch.no_grad():
        assert output.shape[0] == len(target)
        msle = torch.sum(MSELoss(torch.log(output + 1), torch.log(target + 1))).item()
    return msle / len(target)

def rmsle(output, target):
    # root mean square log error
    with torch.no_grad():
        assert output.shape[0] == len(target)
        rmsle = torch.sum(torch.sqrt(MSELoss(torch.log(output + 1), torch.log(target + 1)))).item()
    return rmsle / len(target)
```

### training result

```bash
trainer - INFO -     epoch          : 1
trainer - INFO -     loss           : 0.006033351296714197
trainer - INFO -     mae            : 0.04644732211530209
trainer - INFO -     mape           : 0.7897534126540026
trainer - INFO -     rmse           : 0.0009079981798810573
trainer - INFO -     msle           : 0.00029857408025782205
trainer - INFO -     rmsle          : 0.001693601704862279
trainer - INFO -     val_loss       : 0.001081485502111415
trainer - INFO -     val_mae        : 0.02543863166371981
trainer - INFO -     val_mape       : 0.43928909466663996
trainer - INFO -     val_rmse       : 0.0005114987251193573
trainer - INFO -     val_msle       : 9.465186867843538e-05
trainer - INFO -     val_rmsle      : 0.0010227298852987588
                    .
                    .
                    .
                    .
                    .
                    .
trainer - INFO -     epoch          : 7                           <--- final model
trainer - INFO -     loss           : 0.0003257902914095515
trainer - INFO -     mae            : 0.014013349408904712
trainer - INFO -     mape           : 0.2682989962026477
trainer - INFO -     rmse           : 0.00027973901535733605
trainer - INFO -     msle           : 3.3442232215747935e-05
trainer - INFO -     rmsle          : 0.0005611276037816424
trainer - INFO -     val_loss       : 0.00037339228950440884
trainer - INFO -     val_mae        : 0.014986828123529751
trainer - INFO -     val_mape       : 0.26452333776156106
trainer - INFO -     val_rmse       : 0.00030040072995082785
trainer - INFO -     val_msle       : 4.145120674487164e-05
trainer - INFO -     val_rmsle      : 0.0006239583902060985
trainer - INFO - Saving checkpoint: saved/models/correlation/1031_043742/checkpoint-epoch7.pth ...
trainer - INFO - Saving current best: model_best.pth ...
                    .
                    .
                    .
                    .
                    .
                    .
trainer - INFO -     epoch          : 10                           <--- early stop
trainer - INFO -     loss           : 0.00030936458103436354
trainer - INFO -     mae            : 0.013654345855737725
trainer - INFO -     mape           : 0.2527313926269611
trainer - INFO -     rmse           : 0.0002726934728755926
trainer - INFO -     msle           : 3.264761346902863e-05
trainer - INFO -     rmsle          : 0.0005503339679174435
trainer - INFO -     val_loss       : 0.00048039960558526217
trainer - INFO -     val_mae        : 0.017462029221157232
trainer - INFO -     val_mape       : 0.28779067628582317
trainer - INFO -     val_rmse       : 0.0003408467676490545
trainer - INFO -     val_msle       : 4.472730096798235e-05
trainer - INFO -     val_rmsle      : 0.0006782710350429019
trainer - INFO - Validation performance didn't improve for 2 epochs. Training stops.
```
### testing result
```bash
Loading checkpoint: saved/models/correlation/model_best.pth ...
Done
Testing set samples: 30000
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 59/59 [00:19<00:00,  3.04it/s]
Testing result:
{'loss': 0.0003678673184632013, 'mae': 0.014900806681315104, 'mape': 0.31981751302083333, 'rmse': 3.768852154413859e-05, 'msle': 4.456466436386108e-06, 'rmsle': 8.922487099965414e-05}
```

## recommending stickers

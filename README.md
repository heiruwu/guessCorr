# Piccollage take home quiz

## guess the correlation
### data inspection

![dist](https://github.com/gt212345/piccollage/blob/main/materials/dist.png?raw=true)

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
    (14): Conv2d(32, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (#14): params: (3*3*32+1) * 16 = 4624
    (15): ReLU(inplace=True)
    (16): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (avgpool): AdaptiveAvgPool2d(output_size=(1, 1))
  (linear): Sequential(
    (0): Conv2d(16, 1, kernel_size=(1, 1), stride=(1, 1))
    #(0): params: (16+1) * 1 = 17
    (1): Tanh()
  )
)
Trainable parameters: 46625
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
trainer - INFO -     loss           : 0.006355226918278883
trainer - INFO -     mae            : 0.052923480829844875
trainer - INFO -     mape           : 0.9336253106196721
trainer - INFO -     rmse           : 0.0010376017390129467
trainer - INFO -     msle           : 0.0002965020411356818
trainer - INFO -     rmsle          : 0.0018319533641139667
trainer - INFO -     val_loss       : 0.0016302405109939475
trainer - INFO -     val_mae        : 0.03154845953484376
trainer - INFO -     val_mape       : 0.6093135625521342
trainer - INFO -     val_rmse       : 0.000627685441247498
trainer - INFO -     val_msle       : 0.00011339346442643243
trainer - INFO -     val_rmsle      : 0.00116287796649461
trainer - INFO - Saving checkpoint: saved/models/correlation/1031_043742/checkpoint-epoch1.pth ...
trainer - INFO - Saving current best: model_best.pth ...
trainer - INFO -     epoch          : 2
trainer - INFO -     loss           : 0.0013387667537899687
trainer - INFO -     mae            : 0.02858347536996007
trainer - INFO -     mape           : 0.4554133217533429
trainer - INFO -     rmse           : 0.0005664397936197929
trainer - INFO -     msle           : 0.00011448837239731801
trainer - INFO -     rmsle          : 0.001165762629903232
trainer - INFO -     val_loss       : 0.0011161217279732228
trainer - INFO -     val_mae        : 0.02583611766000589
trainer - INFO -     val_mape       : 0.4519783794085185
trainer - INFO -     val_rmse       : 0.0005190427335134397
trainer - INFO -     val_msle       : 8.991883131238865e-05
trainer - INFO -     val_rmsle      : 0.0010030890396640947
trainer - INFO - Saving checkpoint: saved/models/correlation/1031_043742/checkpoint-epoch2.pth ...
trainer - INFO - Saving current best: model_best.pth ...
trainer - INFO -     epoch          : 3
trainer - INFO -     loss           : 0.0010287843780436865
trainer - INFO -     mae            : 0.025002624535312254
trainer - INFO -     mape           : 0.45268593460321427
trainer - INFO -     rmse           : 0.0004976699336742362
trainer - INFO -     msle           : 9.733292861164955e-05
trainer - INFO -     rmsle          : 0.0010504042086928772
trainer - INFO -     val_loss       : 0.0010174695290625095
trainer - INFO -     val_mae        : 0.024844653218984603
trainer - INFO -     val_mape       : 0.4252873717347781
trainer - INFO -     val_rmse       : 0.000495813836886858
trainer - INFO -     val_msle       : 0.0001021431788103655
trainer - INFO -     val_rmsle      : 0.001071739056225245
trainer - INFO - Saving checkpoint: saved/models/correlation/1031_043742/checkpoint-epoch3.pth ...
trainer - INFO - Saving current best: model_best.pth ...
trainer - INFO -     epoch          : 4
trainer - INFO -     loss           : 0.0008877733419649303
trainer - INFO -     mae            : 0.02316888523908953
trainer - INFO -     mape           : 0.4417370690529545
trainer - INFO -     rmse           : 0.0004623158989318957
trainer - INFO -     msle           : 8.924840704518526e-05
trainer - INFO -     rmsle          : 0.0009964773660952537
trainer - INFO -     val_loss       : 0.0008509158659726382
trainer - INFO -     val_mae        : 0.022542520716786384
trainer - INFO -     val_mape       : 0.40636416447162627
trainer - INFO -     val_rmse       : 0.000453114727512002
trainer - INFO -     val_msle       : 8.47562864049299e-05
trainer - INFO -     val_rmsle      : 0.0009530336720248063
trainer - INFO - Saving checkpoint: saved/models/correlation/1031_043742/checkpoint-epoch4.pth ...
trainer - INFO - Saving current best: model_best.pth ...
trainer - INFO -     epoch          : 5
trainer - INFO -     loss           : 0.0008196783287955137
trainer - INFO -     mae            : 0.022245851091419657
trainer - INFO -     mape           : 0.4379123075157404
trainer - INFO -     rmse           : 0.0004442724630353041
trainer - INFO -     msle           : 8.434383068833995e-05
trainer - INFO -     rmsle          : 0.0009579292373382486
trainer - INFO -     val_loss       : 0.0008507503742196908
trainer - INFO -     val_mae        : 0.022621989741921426
trainer - INFO -     val_mape       : 0.4190376108686129
trainer - INFO -     val_rmse       : 0.00045342504520279667
trainer - INFO -     val_msle       : 8.219568048176976e-05
trainer - INFO -     val_rmsle      : 0.0009298045354274412
trainer - INFO - Saving checkpoint: saved/models/correlation/1031_043742/checkpoint-epoch5.pth ...
trainer - INFO - Saving current best: model_best.pth ...
trainer - INFO -     epoch          : 6
trainer - INFO -     loss           : 0.0007577002605345721
trainer - INFO -     mae            : 0.021370203689982492
trainer - INFO -     mape           : 0.37643182346224785
trainer - INFO -     rmse           : 0.00042734077431183927
trainer - INFO -     msle           : 8.020470671999647e-05
trainer - INFO -     rmsle          : 0.0009332669520517811
trainer - INFO -     val_loss       : 0.0007817290533178796
trainer - INFO -     val_mae        : 0.021446637076636157
trainer - INFO -     val_mape       : 0.3858960991203785
trainer - INFO -     val_rmse       : 0.0004341451229993254
trainer - INFO -     val_msle       : 7.425443484195663e-05
trainer - INFO -     val_rmsle      : 0.0008837215988120685
trainer - INFO - Saving checkpoint: saved/models/correlation/1031_043742/checkpoint-epoch6.pth ...
trainer - INFO - Saving current best: model_best.pth ...
trainer - INFO -     epoch          : 7
trainer - INFO -     loss           : 0.0007002292363128314
trainer - INFO -     mae            : 0.020518597305441897
trainer - INFO -     mape           : 0.3729248049631715
trainer - INFO -     rmse           : 0.00041082684373638283
trainer - INFO -     msle           : 7.605220874726608e-05
trainer - INFO -     rmsle          : 0.0009038085666446326
trainer - INFO -     val_loss       : 0.0006979367795089881
trainer - INFO -     val_mae        : 0.020308566043774286
trainer - INFO -     val_mape       : 0.36255427727103234
trainer - INFO -     val_rmse       : 0.00041034069005399943
trainer - INFO -     val_msle       : 7.28205741761485e-05
trainer - INFO -     val_rmsle      : 0.0008633291375978539
trainer - INFO - Saving checkpoint: saved/models/correlation/1031_043742/checkpoint-epoch7.pth ...
trainer - INFO - Saving current best: model_best.pth ...                      <--- final model
trainer - INFO -     epoch          : 8
trainer - INFO -     loss           : 0.0006540426453575492
trainer - INFO -     mae            : 0.01978776603192091
trainer - INFO -     mape           : 0.35929031869769096
trainer - INFO -     rmse           : 0.0003970772436005063
trainer - INFO -     msle           : 7.339442785299374e-05
trainer - INFO -     rmsle          : 0.0008829678067510638
trainer - INFO -     val_loss       : 0.0008433220069855451
trainer - INFO -     val_mae        : 0.022729317272702852
trainer - INFO -     val_mape       : 0.39152298104763034
trainer - INFO -     val_rmse       : 0.00045134716248139737
trainer - INFO -     val_msle       : 7.236197247645274e-05
trainer - INFO -     val_rmsle      : 0.0008795549387577922
trainer - INFO - Saving checkpoint: saved/models/correlation/1031_043742/checkpoint-epoch8.pth ...
trainer - INFO -     epoch          : 9
trainer - INFO -     loss           : 0.000632916721884006
trainer - INFO -     mae            : 0.019499900032455723
trainer - INFO -     mape           : 0.3781795942932367
trainer - INFO -     rmse           : 0.0003903167201109075
trainer - INFO -     msle           : 7.050714648115294e-05
trainer - INFO -     rmsle          : 0.0008619381140063827
trainer - INFO -     val_loss       : 0.0007570908185249815
trainer - INFO -     val_mae        : 0.021355336184302964
trainer - INFO -     val_mape       : 0.43860252540310224
trainer - INFO -     val_rmse       : 0.0004276736608395974
trainer - INFO -     val_msle       : 6.55279004398229e-05
trainer - INFO -     val_rmsle      : 0.0008259134353138507
trainer - INFO - Saving checkpoint: saved/models/correlation/1031_043742/checkpoint-epoch9.pth ...
trainer - INFO -     epoch          : 10
trainer - INFO -     loss           : 0.0006003035469523941
trainer - INFO -     mae            : 0.018973845115552345
trainer - INFO -     mape           : 0.3371219983200232
trainer - INFO -     rmse           : 0.00038005297066411004
trainer - INFO -     msle           : 6.765326295529424e-05
trainer - INFO -     rmsle          : 0.0008411300062046696
trainer - INFO -     val_loss       : 0.0007291348481861254
trainer - INFO -     val_mae        : 0.020892351942757764
trainer - INFO -     val_mape       : 0.3684447823266188
trainer - INFO -     val_rmse       : 0.00041945329774171114
trainer - INFO -     val_msle       : 6.697251868414848e-05
trainer - INFO -     val_rmsle      : 0.0008344784020446241
trainer - INFO - Validation performance didn't improve for 2 epochs. Training stops.
```
### testing result
```bash
Loading checkpoint: saved/models/correlation/model_best.pth ...
Done
Testing set samples: 30000
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 59/59 [00:18<00:00,  3.16it/s]
Testing result:
{'loss': 0.0006764576567957799, 'mae': 0.020008172607421874, 'mape': 0.42591689453125, 'rmse': 5.110644102096558e-05, 'msle': 8.16218654314677e-06, 'rmsle': 0.00012170817852020264}
```

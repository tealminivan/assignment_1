from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import torch

class NNModel(nn.Module):
  def __init__(self):
    super(NNModel, self).__init__()

    def init_weights(m):
      if isinstance(m, nn.Linear):
          torch.nn.init.xavier_normal_(m.weight)
          m.bias.data.fill_(0.01)

    self.layer = nn.Sequential(
        nn.Conv2d(3, 128, 5, padding=2),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.Conv2d(128, 128, 5, padding=2),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),
        nn.Dropout2d(p=0.25),

        nn.Conv2d(128, 256, 3, padding=1),
        nn.BatchNorm2d(256),
        nn.ReLU(),
        nn.Conv2d(256, 256, 3, padding=1),
        nn.BatchNorm2d(256),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),
        nn.Dropout2d(p=0.25)
        )
    
    self.layer.apply(init_weights)
    
    self.dense_layer = nn.Sequential(
        nn.Linear(256 * 8 * 8, 1024),
        nn.BatchNorm1d(1024),
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(1024, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(512, 10)
        )
    
    self.dense_layer.apply(init_weights)


  def forward(self, x):
    results = self.layer(x)
    results = results.view(-1, 256 * 8 * 8)
    results = self.dense_layer(results)
    return results
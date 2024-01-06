import numpy as np
import torch

inputs = np.array([[73, 67, 43], 
                   [91, 88, 64],
                   [87, 134, 58],
                   [102, 43, 37],
                   [69, 96, 70]], dtype='float32')

targets = np.array([[56, 70], 
                    [81, 101],
                    [119, 133],
                    [22, 37],
                    [103, 119]], dtype='float32')

inputs = torch.from_numpy(inputs)
targets = torch.from_numpy(targets)
print(inputs)
print(targets)

w = torch.randn(2, 3, requires_grad=True)
b = torch.randn(2, requires_grad=True)
print(w)
print(b)


def model(x):
    return x @ w.t() + b

preds = model(inputs)
print(preds)

def mse(t1, t2):
    diff = t1 - t2
    return torch.sum(diff * diff) / diff.numel()

loss = mse(preds, targets)
print(loss)

loss.backward()
w.grad.zero_()
b.grad.zero_()

for i in range(800):
    preds = model(inputs)
    loss = mse(preds, targets)
    loss.backward()
    with torch.no_grad():
        w -= w.grad * 1e-5
        b -= b.grad * 1e-5
        w.grad.zero_()
        b.grad.zero_()
    
preds = model(inputs)
loss = mse(preds, targets)
print(loss)
print(preds)
print(targets)


import torch.nn as nn

inputs = np.array([[73, 67, 43], 
                   [91, 88, 64],
                   [87, 134, 58],
                   [102, 43, 37],
                   [69, 96, 70],
                   [73, 67, 43], 
                   [91, 88, 64],
                   [87, 134, 58],
                   [102, 43, 37],
                   [69, 96, 70],
                   [73, 67, 43], 
                   [91, 88, 64],
                   [87, 134, 58],
                   [102, 43, 37],
                   [69, 96, 70]], dtype='float32')

targets = np.array([[56, 70], 
                    [81, 101],
                    [119, 133],
                    [22, 37],
                    [103, 119],
                    [56, 70], 
                    [81, 101],
                    [119, 133],
                    [22, 37],
                    [103, 119],
                    [56, 70], 
                    [81, 101],
                    [119, 133],
                    [22, 37],
                    [103, 119]], dtype='float32')

inputs = torch.from_numpy(inputs)
targets = torch.from_numpy(targets)
print(inputs)
print(targets)
from torch.utils.data import TensorDataset

train_ds = TensorDataset(inputs, targets)
print(train_ds[0: 3])

from torch.utils.data import DataLoader
batch_size = 5
train_dl = DataLoader(train_ds, batch_size, shuffle=True)

for xb, yb in train_dl:
    print(xb)
    print(yb)

model = nn.Linear(3, 2)
print(model.weight)
print(model.bias)

print(list(model.parameters()))

preds = model(inputs)
print(preds)

import torch.nn.functional as F

loss_fn = F.mse_loss
loss = loss_fn(model(inputs), targets)
print(loss)

opt = torch.optim.SGD(model.parameters(), 1e-5)

def fit(num_epochs, model, loss_fn, opt, train_dl):
    for epoch in range(num_epochs):
        for xb, yb in train_dl:
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            opt.zero_grad()

        if (epoch + 1) % 10 == 0:
            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, loss))

fit(100, model, loss_fn, opt, train_dl)

print(model(inputs))
print(targets)
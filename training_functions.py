"""
Bulk of the code for the nitty-gritty training here.
"""

import numpy as np
import torch
import time


if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')


def lr_decay(optimizer, epoch):
  if epoch % 10 == 0:
    new_lr = learning_rate / (10 ** (epoch // 10))
    optimizer = setlr(optimizer, new_lr)
    print(f'Changed learning rate to {new_lr}')
  return optimizer

def pt_loader(path, eps=1e-6):
  spec = torch.load(path)
  mean = spec.mean()
  std = spec.std()
  spec_norm = (spec - mean) / (std + eps)
  spec_min, spec_max = spec_norm.min(), spec_norm.max()
  spec_scaled = 255 * (spec_norm - spec_min) / (spec_max - spec_min)
  return spec_scaled

def train(model, loss_fn, train_loader, valid_loader, epochs, optimizer,
          train_losses, valid_losses, train_history, accuracy_history, change_lr=None):
  for epoch in range(1,epochs+1):
    start_time = time.time()
    model.train()
    batch_losses=[]
#    if change_lr:
#      optimizer = change_lr(optimizer, epoch)
    for i, data in enumerate(train_loader):
      x, y = data
      x, y = x.to(device), y.to(device)
      optimizer.zero_grad()
      x = x.to(device, dtype=torch.float32)
      y = y.to(device, dtype=torch.long)
      y_hat = model(x)
      loss = loss_fn(y_hat, y)
      loss.backward()
      batch_losses.append(loss.item())
      optimizer.step()
    train_losses.append(batch_losses)
    print(f'Epoch - {epoch} Train-Loss : {np.mean(train_losses[-1])}')
    train_history[epoch] = np.mean(train_losses[-1])
    model.eval()
    batch_losses=[]
    trace_y = []
    trace_yhat = []
    for i, data in enumerate(valid_loader):
      x, y = data
      x = x.to(device, dtype=torch.float32)
      y = y.to(device, dtype=torch.long)
      y_hat = model(x)
      loss = loss_fn(y_hat, y)
      trace_y.append(y.cpu().detach().numpy())
      trace_yhat.append(y_hat.cpu().detach().numpy())
      batch_losses.append(loss.item())
    valid_losses.append(batch_losses)
    trace_y = np.concatenate(trace_y)
    trace_yhat = np.concatenate(trace_yhat)
    accuracy = np.mean(trace_yhat.argmax(axis=1)==trace_y)
    end_time = time.time()
    print(f'Epoch - {epoch} Valid-Loss : {np.mean(valid_losses[-1])} Valid-Accuracy : {accuracy}')
    print('Elapsed time {}s'.format(end_time - start_time))
    accuracy_history[epoch] = accuracy
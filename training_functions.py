"""
Bulk of the code for the nitty-gritty training here.
"""

import numpy as np
import torch
import time
import torch.nn.functional as F


if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')


def lr_decay(learning_rate, optimizer, epoch):
  if epoch % 5 == 0:
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
  if list(spec_scaled.shape) == [1, 401, 61]:
      return spec_scaled
  else:
      spec_padded = F.pad(spec_scaled, (0, 61 - spec_scaled.shape[2]))
      return spec_padded


def train(model, loss_fn, train_loader, valid_loader, epochs, optimizer,
          train_losses, valid_losses, train_history, accuracy_history,
          scheduler):
  start_time = time.time()
  for epoch in range(1,epochs+1):
    model.train()
    batch_losses=[]
    for i, data in enumerate(train_loader):
      if i % 200 == 0:
          print('{} batches completed in training'.format(i))
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
    scheduler.step(np.mean(valid_losses[-1]))
    trace_y = np.concatenate(trace_y)
    trace_yhat = np.concatenate(trace_yhat)
    accuracy = np.mean(trace_yhat.argmax(axis=1)==trace_y)
    end_time = time.time()
    print(f'Epoch - {epoch} Valid-Loss : {np.mean(valid_losses[-1])} Valid-Accuracy : {accuracy}')
    print('Elapsed time {}s'.format(end_time - start_time))
    accuracy_history[epoch] = accuracy

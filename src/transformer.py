
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch
from torch.utils.data import Dataset, random_split
from collections import Counter
import numpy as np
import math
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd

device = 'cuda:0'


def hist(s):
  c = Counter(s)
  c = {w: c[w] for w in c}
  return [c[w] for w in s]

class HistogramDataset(Dataset):
    def __init__(self, seq_len, T, n_samples,seed=42):
        self.seq_len = seq_len
        self.T = T
        self.n_samples = n_samples
        rs = np.random.RandomState(seed)
        self.X = rs.randint(0, T, (n_samples, seq_len))
        self.X = np.unique(self.X, axis=0)
        self.y = np.empty_like(self.X)
        self.n_samples = self.X.shape[0]
        for i in range(self.n_samples):
          self.y[i] = hist(self.X[i])

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx],dtype=torch.long), torch.tensor(self.y[idx],dtype=torch.long)


class DotProductAttention(nn.Module):
  def __init__(self, model_dim,attention_input='both'):
    super(DotProductAttention, self).__init__()

    if attention_input not in ['both','only_sem','only_pos']:
      raise ValueError

    self.attention_input = attention_input

    self.model_dim = model_dim

    if self.attention_input == 'both':
        a = model_dim
    elif self.attention_input == 'only_sem' or self.attention_input == 'only_pos':
        a = int(model_dim/2)

    self.F = torch.zeros(model_dim,model_dim,device=device)

    with torch.no_grad():
      if self.attention_input in ['both','only_sem']:
        self.F[torch.arange(0,a),torch.arange(0,a)] = 1.0
      elif self.attention_input in ['both','only_pos']:
        self.F[torch.arange(a,2*a),torch.arange(a,2*a)] = 1.0

    self.Q = nn.Parameter(torch.empty(model_dim,model_dim,device=device))
    self.K = nn.Parameter(torch.empty(model_dim,model_dim,device=device))
    self.V = nn.Parameter(torch.empty(model_dim, model_dim,device=device))

    nn.init.kaiming_uniform_(self.Q.T, a=math.sqrt(5))
    nn.init.kaiming_uniform_(self.K.T, a=math.sqrt(5))
    nn.init.kaiming_uniform_(self.V.T, a=math.sqrt(5))

    self.attn_probs = None # we want to readout the attention matrix later

  def forward(self, x):
    Qx = x @ self.F @ self.Q
    Kx = x @ self.F @ self.K
    Vx = x @ self.V

    attn_scores = torch.matmul(Qx,Kx.transpose(-2,-1)) / math.sqrt(self.model_dim)
    attn_probs = torch.softmax(attn_scores,dim=-1)
    x = torch.matmul(attn_probs,Vx)

    self.attn_probs = attn_probs
    return x



class LearnedPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super(LearnedPositionalEncoding, self).__init__()

        pe = torch.arange(0, max_seq_length)
        self.embedding = nn.Embedding(max_seq_length, d_model)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        e = self.embedding(self.pe)
        return torch.tile(e,(x.shape[0], 1, 1))


class TransformerSeq2Seq(nn.Module):

  def __init__(self,T,model_dim,p,n_classes,L,attention_input):
    super(TransformerSeq2Seq, self).__init__()
    if model_dim % 2 == 1:
      raise ValueError()

    self.model_dim = model_dim
    self.attention_input = attention_input
    embedding_dim = model_dim

    self.semantic_emb = nn.Embedding(T, embedding_dim)
    self.positional_emb = LearnedPositionalEncoding(embedding_dim,L)
    self.attention = DotProductAttention(model_dim,attention_input=attention_input)
    self.norm = nn.LayerNorm(model_dim)
    self.fc1 = nn.Linear(model_dim, p)
    self.activ = nn.ReLU()
    self.fc2 = nn.Linear(p, n_classes)

  def forward(self,x): # B x L
    x_sem = self.semantic_emb(x) # B x L x d/2 or B x L x d
    x_pos = self.positional_emb(x)
    if self.attention_input in ['only_sem', 'only_pos']:
        x_sem[...,int(self.model_dim/2):] = 0.0
        x_pos[...,:int(self.model_dim/2)] = 0.0
    x = x_sem + x_pos
    a = self.attention(x)
    x = self.norm(a) # +x
    x = self.fc2(self.activ(self.fc1(x)))
    return x


def train(transformer, train_dataset, val_dataset,n_epochs = 100, n_classes = 10):

  lr = 0.001


  train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
  val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=True)

  criterion = nn.CrossEntropyLoss()
  optimizer = optim.Adam(transformer.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)

  train_losses = []
  val_losses = []
  val_acc = []

  for epoch in range(n_epochs):

          epoch_loss = 0.0

          for X, y in train_dataloader:
              X = X.to(device)
              y = y.to(device)
              optimizer.zero_grad()

              output = transformer(X)
              loss = criterion(output.contiguous().view(-1, n_classes), y.contiguous().view(-1))

              epoch_loss += loss.item()

              loss.backward()
              optimizer.step()

          epoch_loss /= len(train_dataloader)
          train_losses.append(epoch_loss)

          # Evaluate on the test set every epoch
          with torch.no_grad():
              val_loss = 0.0
              acc = 0.0
              for X, y in val_dataloader:
                  X = X.to(device)
                  y = y.to(device)
                  output = transformer(X)
                  pred = output.argmax(axis=-1)
                  loss = criterion(output.view(-1,n_classes), y.view(-1))
                  acc += torch.mean((pred.view(-1)==y.view(-1)).float()).item()
                  val_loss+=loss.item()
              val_loss /= len(val_dataloader)
              acc /= len(val_dataloader)

              val_losses.append(val_loss)
              val_acc.append(acc)

          if epoch % 10 == 0:
            print(f'[Epoch {epoch:02}] Train loss = {epoch_loss:.5f} :: Val loss {val_loss:.5f} :: Val accuracy {acc*100:.2f}')
  return transformer, train_losses, val_losses, val_acc


def predict_sequence(x,transformer):
  x_ = torch.tensor(x,dtype=torch.long,device=device).unsqueeze(0)
  pred = transformer(x_).argmax(axis=-1).detach().cpu().numpy()
  y = hist(x)
  return list(pred), np.all(y == pred)


import copy

def reparameterize(orig_transformer,T,model_dim,p,n_classes,L):
  with torch.no_grad():
    a = orig_transformer.state_dict()
    new_transformer = TransformerSeq2Seq(T,model_dim,p,n_classes,L,orig_transformer.attention_input)
    new_transformer.load_state_dict(a)
    new_transformer.attention.Q.data = new_transformer.attention.F @ new_transformer.attention.Q
    new_transformer.attention.K.data = new_transformer.attention.F @ new_transformer.attention.K
    new_transformer.semantic_emb.weight[...,int(model_dim/2):] = 0.0
    new_transformer.positional_emb.embedding.weight[...,:int(model_dim/2)] = 0.0
    # add some small random noise to Q and K, and the embedding weights
    new_transformer.attention.Q.data += 0.001*torch.randn_like(new_transformer.attention.Q.data)
    new_transformer.attention.K.data += 0.001*torch.randn_like(new_transformer.attention.K.data)
    new_transformer.semantic_emb.weight.data += 0.001*torch.randn_like(new_transformer.semantic_emb.weight.data)
    new_transformer.positional_emb.embedding.weight.data += 0.001*torch.randn_like(new_transformer.positional_emb.embedding.weight.data)
    a = new_transformer.state_dict()
    new_transformer = TransformerSeq2Seq(T,model_dim,p,n_classes,L,'both')
    new_transformer.load_state_dict(a)
  return new_transformer


def train_local(transformer, train_dataset, val_dataset,n_epochs,n_classes):

  lr = 0.001
  

  train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
  val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=True)

  criterion = nn.CrossEntropyLoss()
  optimizer = optim.SGD(transformer.parameters(), lr=lr)
  #optimizer = optim.Adam(transformer.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)

  train_losses = []
  val_losses = []
  val_acc = []

  for epoch in range(n_epochs):

          epoch_loss = 0.0

          for X, y in train_dataloader:
              X = X.to(device)
              y = y.to(device)
              optimizer.zero_grad()

              output = transformer(X)
              loss = criterion(output.contiguous().view(-1, n_classes), y.contiguous().view(-1))

              epoch_loss += loss.item()

              loss.backward()
              optimizer.step()

          epoch_loss /= len(train_dataloader)
          train_losses.append(epoch_loss)

          # Evaluate on the test set every epoch
          with torch.no_grad():
              val_loss = 0.0
              acc = 0.0
              for X, y in val_dataloader:
                  X = X.to(device)
                  y = y.to(device)
                  output = transformer(X)
                  pred = output.argmax(axis=-1)
                  loss = criterion(output.view(-1,n_classes), y.view(-1))
                  acc += torch.mean((pred.view(-1)==y.view(-1)).float()).item()
                  val_loss+=loss.item()
              val_loss /= len(val_dataloader)
              acc /= len(val_dataloader)

              val_losses.append(val_loss)
              val_acc.append(acc)

          if epoch % 10 == 0:
            print(f'[Epoch {epoch:02}] Train loss = {epoch_loss:.5f} :: Val loss {val_loss:.5f} :: Val accuracy {acc*100:.2f}')
  return transformer, train_losses, val_losses, val_acc

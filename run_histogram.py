# ---
# # Training a single Layer of Attention on the Histogram Task may lead to two solutions
# 
# This notebook shows how to train and evaluate single-layer transformers with dot-product attention and trained positional encodings.
# 
# - Part 1: Define data and model architecture
# - Part 2: Training of models with different settings on the attention layer (positional, semantic, or both)
# - Part 3: Introspecting the attention layers for some input sequences
# - Part 4: Checking whether the parameter values with the frozen weights stay close to their original position in the unfrozen weight space

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
from src.transformer import *

save_dir = Path('empirics/histogram')
save_dir.mkdir(exist_ok=True, parents=True)
device = 'cuda:0'

# Define the length, maximum value, and number of samples
seq_len = 10
T = 15
num_samples = 50000
n_classes = seq_len+1

# Create the dataset
dataset = HistogramDataset(seq_len, T, num_samples)

# Split the dataset into train, test, and validation sets
train_ratio, val_ratio = 0.7, 0.3

train_size = int(train_ratio * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_size, val_size

p = 128
model_dim = 64
n_classes = seq_len+1
L = seq_len
n_runs = 10
model_types = ['only_sem','only_pos']
n_epochs = 200


results = []

for model_type in model_types:
    print(f'Running {model_type}...')
    for i in range(n_runs):
        transformer = TransformerSeq2Seq(T,model_dim,p,n_classes,L,model_type).to(device)
        torch.save(transformer.state_dict(),save_dir / f'run_{i}_initmodel_{transformer.attention_input}_orig.pt')
        transformer, train_losses, val_losses, val_acc = train(transformer, train_dataset, val_dataset,n_epochs=n_epochs,n_classes=n_classes)
        torch.save(transformer.state_dict(),save_dir / f'run_{i}_model_{transformer.attention_input}_orig.pt')
        results.append({
            'model_type': model_type,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'val_acc': val_acc,
            'run':i,
        })
    print(f'Done.')
    
pd.DataFrame(results).to_csv(save_dir / 'frozen_transformer_result.csv',index=False)


n_epochs = 100
reparameterized_transformers = []

for r in range(n_runs):
  for model_type in model_types:
  
    print(r,model_type)
    orig_trans = TransformerSeq2Seq(T,model_dim,p,n_classes,L,model_type).to(device)
    orig_dict = torch.load(save_dir / f'run_{r}_model_{model_type}_orig.pt')
    orig_trans.load_state_dict(orig_dict)
    
    rep_trans = reparameterize(orig_trans,T,model_dim,p,n_classes,L).to(device)
    torch.save(rep_trans.state_dict(),save_dir / f'run_{r}_model_{model_type}_repar.pt')
    
    rep_trans, train_losses, val_losses, val_acc = train_local(rep_trans, train_dataset, val_dataset,n_epochs,n_classes)
    reparameterized_transformers.append({
      'train_losses': train_losses,
      'val_losses': val_losses,
      'val_acc': val_acc,
      'run': r,
      'model_type': model_type,
    })
    torch.save(rep_trans.state_dict(),save_dir /f'run_{r}_model_{model_type}_retrained.pt')
    
pd.DataFrame(reparameterized_transformers).to_csv(save_dir / 'reparameterized_transformers.csv',index=False)

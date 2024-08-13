import torch
import torch.nn as nn
from torch.nn import functional as F

import numpy as np
 

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f'Using {device} device')

print("\n-------------------\n")

torch.manual_seed(1337)

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

print(f'Length of the text file: {len(text)}')

chars = sorted(list(set(text)))
vocab_size = len(chars)
print(''.join(chars))
print(vocab_size)

stoi = {ch:i for i, ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}
encode = lambda s : [stoi[ch] for ch in s]
decode = lambda l : ''.join(itos[i] for i in l)

print(encode('hello'))
print(decode(encode('hello')))

#Sentence Piece
#tiktoken 

print("\n-------------------\n")

data = torch.tensor(encode(text), dtype=torch.long)
print(data.shape, data.dtype)

n = int(0.9*len(data))
train_data, val_data = data[:n], data[n:]

block_size = 8

x = train_data[:block_size]
y = train_data[1:block_size+1]

for t in range(block_size):
    context = x[:t+1]
    target = y[t]
    print(f'When input is {context} the target is {target}')

batch_size = 4
block_size = 8

def get_batch(split):
    data = train_data if split == 'train' else val_data
    start_idx = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[idx:idx+block_size] for idx in start_idx]).to(device)
    y = torch.stack([data[idx+1:idx+block_size+1] for idx in start_idx]).to(device)
    return x, y

print("\n-------------------\n")

xb, yb = get_batch('train')
print('inputs: ', xb.shape, xb.dtype)
print('targets: ', yb.shape, yb.dtype)

for b in range(batch_size):
    for t in range(block_size):
        context = xb[b, :t+1]
        target = yb[b, t]
        print(f'Batch {b}, when input is {context} the target is {target}')
    break


print("\n-------------------\n")

class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, target=None):
        logits = self.token_embedding_table(idx)

        if target is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            target = target.view(-1)

            loss = F.cross_entropy(logits, target)

        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):

            logits, loss = self(idx)
            logits = logits[:,-1,:]
            prob = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(prob, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)
        return idx

model = BigramLanguageModel(vocab_size).to(device)
logits, loss = model(xb, yb)
print(logits.shape, xb.shape, loss)
print(decode(model.generate(idx=torch.zeros((1,1), dtype=torch.long, device=device), max_new_tokens=100)[0].tolist()))

print("\n-------------------\n")

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

bacth_size = 32

for steps in range(1000):
    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print('Final Loss: {:.3f}'.format(loss.item()))
print(decode(model.generate(idx=torch.zeros((1,1), dtype=torch.long, device=device), max_new_tokens=500)[0].tolist()))


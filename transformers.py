import torch
import torch.nn as nn
from torch.nn import functional as F

torch.manual_seed(1337)

#######################################################
#hyperparameters
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dropout = 0

batch_size = 64
block_size = 256
n_embed = 384
n_head = 6
n_layer = 6
dropout = 0.2

learning_rate = 3e-4
max_iterations = 5000
eval_iterations = 500
eval_interval = 200

#######################################################

class LayerNorm1D(nn.Module):
    '''
    Custom Layer Normalization class for 1D data
    x_hat = gamma * (x - mean) / sqrt(var + eps) + beta
    '''
    def __init__(self, n_embed, eps=1e-5, momentum=0.1):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(n_embed))
        self.beta = nn.Parameter(torch.zeros(n_embed))

    def forward(self, x):   
        xmean = x.mean(1, keepdim=True)
        x_var = x.var(1, keepdim=True)
        x_hat = (x - xmean) / torch.sqrt(x_var + self.eps)
        self.out = self.gamma * x_hat + self.beta
        return self.out
        
class Head(nn.Module):
    '''
    Single Head of Multi-head Self-Attention
    Contains:
         -Key, Query and Value Parameters
    Parameters:
        n_embed     - Size of the input embedding
        head_size   - Size of the key, query and value vectors
        block_size  - Max sequence length - Length of the Attention Block
        encoder     - Boolean flag to determine if the head is in the encoder or decoder
    Inputs:
        x           - Input Embeddings, shape (Batch, Context Length, Embedding dimension)
    Outputs:
        out         - Context Aware Embeddings
    '''

    def __init__(self, n_embd, head_size, block_size, encoder=False):
        super().__init__()

        self.encoder = encoder

        #Linear layers for key, query and value. Note: Bias is set to False
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)

        #Non-learnable parameter for masking purposes
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size))) 

        #Attention map
        self.attention_map = None

        #Dropout layer
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        B, T, C = x.shape
        
        k = self.key(x)
        q = self.query(x)

        #W = Q . K / sqrt(dk) - Division to ensure numerical stability
        # ( B, T, C) @ (B, C, T) = (B, T, T)
        weight = q @ k.transpose(-2, -1) * C**(-0.5)

        #Masked Attention for decoder - Cannot look at future tokens
        if not self.encoder:
            weight = torch.masked_fill(weight, self.tril[:T, :T] == 0, float('-inf'))
        
        weight = F.softmax(weight, dim=-1)
        weight = self.dropout(weight)

        #For visualization purposes
        self.attention_map = weight

        v = self.value(x)

        #(B, T, T) @ (B, T, C) = (B, T, C)
        out = weight @ v
        
        return out

class MultiHeadAttention(nn.Module):
    '''
    Multi-head self-attention block
    Contains:
        - Multiple Heads to focus on different aspects
        - Projection layer for the output
    Parameters:
        num_heads   - Number of heads in the block
        n_embed     - Size of the input embedding
        head_size   - Size of the key, query and value vectors
        block_size  - Max sequence length - Length of the Attention Block
        encoder     - Boolean flag to determine if the head is in the encoder or decoder
    Inputs:
        x           - Input Embeddings, shape (Batch, Context Length, Embedding dimension)
    Outputs:
        out         - Context Aware Embeddings
    '''
    def __init__(self, num_heads, n_embd, head_size, block_size, encoder=False):
        super().__init__()
        self.heads = nn.ModuleList([Head(n_embed, head_size, block_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        scores = torch.cat([h(x) for h in self.heads], dim=-1)
        scores = self.proj(scores)
        out = self.dropout(scores)

        return out
    
class FeedForward(nn.Module):
    '''
    Feed Forward Layers to add non-linearity
    Parameters:
        n_embed     - Size of the input embedding
    Inputs:
        x           - Multi-head attention output, shape (Batch, Context Length, Embedding dimension)
    Outputs:
        out         - Context Aware Embeddings
    '''
    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4*n_embed), 
            nn.ReLU(), 
            nn.Linear(4*n_embed, n_embed),
            nn.ReLU(), 
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)
    
class Block(nn.Module):
    '''
    Block of Multi-Head Self-Attention
    Contains:
        - Layer Normalization (Pre-layer normalization used in this case) 
        - Multi-Head Self-Attention
        - Residual Connections
        - Feed Forward Layer
    LayerNorm -> MultiHeadAttention -> Residual Connection -> LayerNorm -> FeedForward -> Residual Connection
    Parameters:
        num_heads   - Number of heads in the block
        n_embed     - Size of the input embedding
        block_size  - Max sequence length - Length of the Attention Block
        encoder     - Boolean flag to determine if the head is in the encoder or decoder
    Inputs:
        x           - Input Embeddings, shape (Batch, Context Length, Embedding dimension)
    Outputs:
        out         - Context Aware Embeddings
    '''
    def __init__(self, num_heads, n_embd, block_size, encoder=False):
        super().__init__()

        #Head Size is the embedding size divided by the number of heads
        head_size = n_embed // n_head
        self.sa = MultiHeadAttention(num_heads, n_embd, head_size, block_size, encoder)
        self.ff = FeedForward(n_embed)

        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x
    
class Decoder(nn.Module):
    '''
    Decoder Block
    Contains:
        - Multiple Blocks of Multi-Head Self-Attention
        - Masked self-attention
    Parameters:
        vocab_size  - Size of the vocabulary
        n_layers    - Number of Blocks in the decoder
        n_embed     - Size of the input embedding
        n_head      - Number of heads in each block
        block_size  - Max sequence length - Length of the Attention Block
    Inputs:
        x           - Input Data, shape (Batch, Context Length, Embedding dimension)
    Outputs:
        out         - Context Aware Embeddings
    '''    
    def __init__(self, vocab_size, n_layers, n_embed, n_head, block_size):
        super().__init__()
        
        #Embedding layers - includes positional encoding
        self.token_embedding = nn.Embedding(vocab_size, n_embed)
        self.positional_embedding = nn.Embedding(block_size, n_embed)

        #Blocks of Multi-Head Self-Attention
        self.blocks = nn.Sequenial(*[Block(n_head, n_embed, block_size) for _ in range(n_layers)])

        #Layer Normalization
        self.ln = nn.LayerNorm(n_embed)

        #Linear Layer to transform embeddings to logits
        self.linear = nn.Linear(n_embed, vocab_size)

    def forward(self, indices, targets=None):
        B, T = indices.shape

        #Embeddings
        tok_embed = self.token_embedding(indices)
        pos_embed = self.positional_embedding(torch.arange(T, device=device))

        x = tok_embed + pos_embed
        out = self.blocks(x)
        out = self.ln(out)
        logits = self.linear(out)

        B, T, v_size = logits.shape

        attention_maps = []
        for block_id, block in enumerate(self.blocks):
            for head_id, head in enumerate(block.sa.heads):
                attention_maps.append(head.attention_map)
        
        return logits
    
    def generate(self, idx, max_new_tokens):
        #Auto-regressive
        #Generate the next token based on a sequence of previous tokens (max sequence length is given by block_size)
        #Feed them back into the model to predic the next token
        #Iterate this process to generate a sequence of max new tokens

        for _ in range(max_new_tokens):
            idx_updated = idx[:,-block_size,:]
            logits = self(idx_updated)

            #Using only the last token in the sequence
            logits = logits[:,-1,:]
            prob = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(prob, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)

        return idx

class GPTLanguageModel(nn.Module):
    '''
    Bigram Language Model
    Contains:
        - Decoder Block
    '''
    def __init__(self, vocab_size, n_layers, n_embed, n_head, block_size):
        super().__init__()

        #Embedding layers - includes positional encoding
        self.token_embedding = nn.Embedding(vocab_size, n_embed)
        self.positional_embedding = nn.Embedding(block_size, n_embed)
        
        #Blocks of Multi-Head Self-Attention
        self.blocks = nn.Sequential(*[Block(n_head, n_embed, block_size, encoder=False) for _ in range(n_layers)])

        #Layer Normalization
        self.ln = nn.LayerNorm(n_embed)

        #Linear Layer to transform embeddings to logits
        self.linear = nn.Linear(n_embed, vocab_size)

    def forward(self, x, target=None):
        B, T= x.shape

        tok_embed = self.token_embedding(x)
        pos_embed = self.positional_embedding(torch.arange(T, device=device))

        x = tok_embed + pos_embed
        out = self.blocks(x)
        out = self.ln(out)
        logits = self.linear(out)

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
            idx_cond = idx[:,-block_size, :]
            logits, _ = self(idx)
            logits = logits[:, -1, :]
            prob = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(prob, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)

        return idx
    

print("\n-------------------\n\n")

print(f'Using {device} device')
print("\n-------------------\n")

#Loading the data
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = set(text)
vocab_size = len(chars)

#Mapping from character to index and vice-versa
stoi = {ch : i for i, ch in enumerate(chars)}
itos = {i : ch for i, ch in enumerate(chars)}

encode = lambda x : [stoi[ch] for ch in x]
decode = lambda x : ''.join([itos[i] for i in x])

#Train and Validation data
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))

train_data = data[:n]
val_data = data[n:]

#Get a batch of data
def get_batch(split):
    data = train_data if split == 'train' else val_data
    start_idx = torch.randint(len(data) - block_size, (batch_size,))
    
    x = torch.stack([data[idx:idx+block_size] for idx in start_idx]).to(device)
    y = torch.stack([data[idx+1:idx+block_size+1] for idx in start_idx]).to(device)
    
    return x, y

#Loss Estimation
@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()

    for split in ['train', 'val']:
        losses = torch.zeros(eval_iterations)
        for k in range(eval_iterations):
            x, y = get_batch(split)
            logits, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()

    model.train()

    return out

model = GPTLanguageModel(vocab_size, n_layer, n_embed, n_head, block_size)
model.to(device)

print(f'Number of parameters in the GPT Language Model are:{sum(p.numel() for p in model.parameters())/1e6}M parameters')
print("\n-------------------\n\n")

#Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for iter in range(max_iterations):

    if iter % eval_interval == 0 or iter == max_iterations-1:
        losses = estimate_loss(model)
        print(f'Iteration: {iter}, Train Loss: {losses["train"]}, Validation Loss: {losses["val"]}')

    xb, yb = get_batch('train')

    logits, loss = model(xb, yb)

    #Clear the gradients
    optimizer.zero_grad(set_to_none=True)

    #Backpropagation
    loss.backward()

    #Update the weights
    optimizer.step()

context = torch.zeros((1,1), dtype=torch.long, device=device)
print(decode(model.generate(idx=context, max_new_tokens=500)[0].tolist()))
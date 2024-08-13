import torch
import torch.nn as nn
from torch.nn import functional as F

torch.manual_seed(1337)

#######################################################
#hyperparameters
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dropout = 0

batch_size = 16
block_size = 8
n_embed = 64
head_size = 16
n_head = 4
n_layer = 4

learning_rate = 1e-3
max_iterations = 5000
eval_iterations = 100

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

    def foward(self, x):
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
    
class Encoder(nn.Module):
    '''
    Encoder Block
    Contains:
        - Multiple Blocks of Multi-Head Self-Attention
        - Unmasked self-attention
    Parameters:
        vocab_size  - Size of the vocabulary
        n_layers    - Number of Blocks in the encoder
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
        self.blocks = nn.Sequential(*[Block(n_head, n_embed, block_size) for _ in range(n_layers)])

        #Layer Normalization
        self.ln = nn.LayerNorm(n_embed)

        #Linear Layer to transform embeddings to logits
        self.linear = nn.Linear(n_embed, vocab_size)

    def forward(self, indices):
        B, T = indices.shape

        #Embeddings
        tok_embed = self.token_embedding(indices)
        pos_embed = self.positional_embedding(torch.arange(T, device=device))

        x = tok_embed + pos_embed
        out = self.blocks(x)
        out = self.ln(out)
        logits = self.linear(out)

        B, T, v_size = logits.shape

        #Can take the mean of the logits to get the final output or use just the last logit in the sequence
        logits = torch.mean(logits, dim=1)

        attention_maps = []
        for block_id, block in enumerate(self.blocks):
            for head_id, head in enumerate(block.sa.heads):
                attention_maps.append(head.attention_map)

        return logits

class ClassifierFeedForward(nn.Module):
    '''
    Feed Forward Layers for the purpose of classification
    Comes after the Encoder Block
    Adds non-linearity to the model
    Parameters:
        n_input     - Size of the input embedding - Matches the embedding size of the transformer
        n_hidden    - Size of the hidden layer
        n_output    - Size of the output layer - Number of classes
    Inputs:
        x           - Output from the Encoder Block, shape (Batch, Embedding dimension)
    Outputs:
        out         - Class predictions
    '''
    def __init__(self, n_input, n_hidden, n_output):
        super().__init__()

        self.fc_layer = nn.Linear(n_input, n_hidden)
        self.output_layer = nn.Linear(n_hidden, n_output)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc_layer(x))
        out = self.output_layer(out)

        return out

class ClassifierModel(nn.Module):
    '''
    Classifier Model - to predict the class of the input eg: sentinment analysis
    Contains:
        - Encoder Block - Unmasked multi-head self-attention
        - Classifier Feed Forward Layers
    Parameters:
        vocab_size  - Size of the vocabulary
        n_embed     - Size of the input embedding
        n_head      - Number of heads in each block
        block_size  - Max sequence length - Length of the Attention Block
        n_layers    - Number of Blocks in the encoder
        n_input     - Size of the input embedding - Matches the embedding size of the transformer
        n_hidden    - Size of the hidden layer
        n_output    - Size of the output layer - Number of classes
    Inputs:
        x          - Input data

    '''
    def __init__(self, vocab_size, n_embed, n_head, block_size, n_layers, n_input, n_hidden, n_output):
        super().__init__()
        self.encoder = Encoder(vocab_size, n_layers, n_embed, n_head, block_size)
        self.classifier = ClassifierFeedForward(n_embed, n_hidden, n_output)

    def forward(self, x):
        context_aware_embeddings = self.encoder(x)
        out = self.classifier(context_aware_embeddings)
        return out

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

class BigramLanguageModel(nn.Module):
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
        self.blocks = nn.Sequential(*[Block(n_embed, n_head) for _ in range(n_layers)])

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
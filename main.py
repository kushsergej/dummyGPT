import torch
import torch.nn as nn
from torch.nn import functional as F



# variables
batch_size = 32                  # how many independent sequences we gonna process in parallel
block_size = 8                   # how many context characters we gonna use for prediction
n_embd = 32
max_iters = 5000
eval_interval = 500
eval_iters = 200
learning_rate = 0.001

torch.manual_seed(1337)
print('---')



# Analyze dataset for vocab_size
with open('dataset.txt', 'r', encoding='utf-8') as file:
    text = file.read()

chars = sorted(list(set(text)))     # unique characters in dataset
vocab_size = len(chars)
print(f'>>> Total chars in dataset is: {vocab_size}')
print('---')



# Implement encoder and decoder
str_to_i = {}
i_to_str = {}
for i, ch in enumerate(chars):
    str_to_i[ch] = i
    i_to_str[i] = ch

def encode(message: str):
    result = []
    for c in message:
        result.append(str_to_i[c])
    return result

def decode(message: list[int]):
    result = ''
    for i in message:
        result += i_to_str[i]
    return result

print(f'>>> {decode(encode('Лев Толстой'))} <--> {encode('Лев Толстой')}')
print('---')



# Encode the dataset and store it into torch.tensor
data = torch.tensor(encode(text), dtype=torch.long)

# Split dataset into train (90%) and validation (10%) sets
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]
print('---')



# data loading
def get_batch(split: str):
    if split == 'train':
        data = train_data
    else:
        data = val_data
    # generate a small batch of data of inputs x and targets y
    ix = torch.randint(len(data) - block_size, (batch_size,))   # create 8 (batch_size) tensors of 32 (block_size) characters length on random positions in data
    x = []
    y = []
    for i in ix:
        x.append(data[i:i+block_size])
        y.append(data[i+1:i+block_size+1])
    return torch.stack(x), torch.stack(y)

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(split)
            logits, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

print('---')




# The Head class implements a single self-attention (sa) head, a core component of the transformer architecture.
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        # Linear layers to project input embeddings into key, query, and value vectors.
        # These are used to compute attention scores and weighted values.
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        # Register a lower-triangular matrix (tril) as a buffer to enforce causality (no peeking ahead).
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
    
    def forward(self, x):
        # x: input tensor of shape (Batch, Time, Channels)
        B, T, C = x.shape
        # Project input to key, query, and value vectors
        k = self.key(x)   # (B, T, head_size)
        q = self.query(x) # (B, T, head_size)
        v = self.value(x) # (B, T, head_size)
        # Compute attention scores ("affinities") between queries and keys
        wei = q @ k.transpose(-2, -1) * C**-0.5  # (B, T, T)
        # Mask out future positions to preserve autoregressive property
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        # Convert scores to probabilities
        wei = F.softmax(wei, dim=-1)
        # Weighted sum of value vectors, according to attention weights
        out = wei @ v  # (B, T, head_size)
        return out



# The BigramLanguageModel is a simple neural network for language modeling (subclass of nn.Module)
# The BigramLanguageModel class defines a simple transformer-based language model.
class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Embedding layer for tokens: maps each token index to a vector of size n_embd
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        # Embedding layer for positions: adds positional information to each token
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        # Single self-attention head (Head class) to process the embeddings
        self.sa_head = Head(n_embd)
        # Linear layer to project the output of the attention head to vocabulary logits
        self.lm_head = nn.Linear(n_embd, vocab_size)
    
    def forward(self, idx, targets=None):
        # idx: (batch, time) tensor of token indices
        B, T = idx.shape
        # Get token embeddings for each token in the input
        token_embd = self.token_embedding_table(idx)  # (B, T, n_embd)
        # Get position embeddings for each position in the sequence
        pos_embd = self.position_embedding_table(torch.arange(T))  # (T, n_embd)
        # Add token and position embeddings
        x = token_embd + pos_embd  # (B, T, n_embd)
        # Pass through the self-attention head
        x = self.sa_head(x)  # (B, T, n_embd)
        # Project to logits for each token in the vocabulary
        logits = self.lm_head(x)  # (B, T, vocab_size)

        # If targets are provided, compute the cross-entropy loss
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        # Autoregressively generate new tokens, one at a time
        for _ in range(max_new_tokens):
            # Only use the last block_size tokens as context
            idx_cond = idx[:, -block_size:]
            # Get logits from the model
            logits, loss = self(idx_cond)
            # Focus on the logits for the last time step
            logits = logits[:, -1, :]    # (batch, vocab_size)
            # Convert logits to probabilities
            probs = F.softmax(logits, dim=-1)
            # Sample the next token from the probability distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # Append the new token to the sequence
            idx = torch.cat((idx, idx_next), dim=1)
        return idx



# Instantiate the model with the vocabulary size.
model = BigramLanguageModel()

# Cretae a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f'>>> Step {iter}: train loss {losses['train']}, val loss {losses['val']}')

    # Get a batch of training data: input tokens (xb) and target tokens (yb).
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


context = torch.zeros((1, 1), dtype=torch.long)
print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))
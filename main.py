import torch
import torch.nn as nn
from torch.nn import functional as F
from dotenv import load_dotenv

load_dotenv()  # load environment variables from .env



# variables
batch_size = 64                  # how many independent sequences we gonna process in parallel
block_size = 256                   # how many context characters we gonna use for prediction
n_embd = 384
max_iters = 5000
eval_interval = 500
eval_iters = 200
learning_rate = 0.0003
n_head = 6
n_layer = 6
dropout = 0.2

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




class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)   # (B, T, head_size)
        q = self.query(x) # (B, T, head_size)
        v = self.value(x) # (B, T, head_size)
        wei = q @ k.transpose(-2, -1) * C**-0.5  # (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        out = wei @ v  # (B, T, head_size)
        return out



class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return out



class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4*n_embd),
            nn.ReLU(),
            nn.Linear(4*n_embd, n_embd),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)



class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
    
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x



class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        blocks = []
        for _ in range(n_layer):
            blocks.append(Block(n_embd, n_head=n_head))
        self.blocks = nn.Sequential(*blocks)
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
    
    def forward(self, idx, targets=None):
        B, T = idx.shape
        token_embd = self.token_embedding_table(idx)  # (B, T, n_embd)
        pos_embd = self.position_embedding_table(torch.arange(T))  # (T, n_embd)
        x = token_embd + pos_embd  # (B, T, n_embd)
        x = self.blocks(x)
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
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]    # (batch, vocab_size)
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
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
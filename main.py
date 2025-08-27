import torch
import torch.nn as nn
from torch.nn import functional as F



# variables
batch_size = 32                  # how many independent sequences we gonna process in parallel
block_size = 8                   # how many context characters we gonna use for prediction
n_embd = 32
max_iters = 3000
eval_interval = 300
eval_iters = 200
learning_rate = 0.01

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



# The BigramLanguageModel is a simple neural network for language modeling (subclass of nn.Module)
# It predicts the next token in a sequence based only on the current token (bigram model).
class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Embedding layer that maps each token (character) index to a vector of size n_embd.
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        # Embedding layer that maps each position in the input sequence (up to block_size) to a vector of size n_embd.
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        # Linear layer that projects the embedding vectors back to vocabulary size for prediction.
        self.lm_head = nn.Linear(n_embd, vocab_size)
    
    def forward(self, idx, targets=None):
        # B = batch size, T = sequence length (number of tokens in each input)
        B, T = idx.shape
        # Get token embeddings for each token in the input batch (shape: [B, T, n_embd])
        token_embd = self.token_embedding_table(idx)
        # Get position embeddings for each position in the sequence (shape: [T, n_embd])
        pos_embd = self.position_embedding_table(torch.arange(T))
        # Add token and position embeddings to inject order information (shape: [B, T, n_embd])
        x = token_embd + pos_embd
        # Project embeddings to vocabulary size to get logits for each token position (shape: [B, T, vocab_size])
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            # Reshape logits and targets to be 2D tensors for cross-entropy loss calculation.
            # logits: (B, T, C) -> (B*T, C), targets: (B, T) -> (B*T)
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            # Compute the cross-entropy loss between the predicted logits and the actual targets.
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            # Get the model predictions (logits) for the current input sequence.
            logits, loss = self(idx)
            # Only use the last token in the sequence for prediction.
            logits = logits[:, -1, :]    # Focus on the last time step's logits for each batch
            # Convert logits to probabilities using softmax.
            probs = F.softmax(logits, dim=-1)
            # Sample the next token from the probability distribution.
            idx_next = torch.multinomial(probs, num_samples=1)
            # Append the sampled token to the input sequence.
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
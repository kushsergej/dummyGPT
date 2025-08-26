from turtle import forward
from sympy import Idx
import torch
from torch._functorch.vmap import out_dims_t
import torch.nn as nn
from torch.nn import functional as F



# Analyze dataset for vocab_size
with open('dataset.txt', 'r', encoding='utf-8') as file:
    text = file.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
print(f'Total chars in dataset is: {vocab_size}')



# Implement encoder and decoder
str_to_i = {}
i_to_str = {}
for i, ch in enumerate(chars):
    str_to_i[ch] = i
    i_to_str[i] = ch

def encode(message: str) -> list[int]:
    result = []
    for c in message:
        result.append(str_to_i[c])
    return result

def decode(message: list[int]) -> str:
    result = ''
    for i in message:
        result += i_to_str[i]
    return result

print(f'{decode(encode('Лев Толстой'))} <--> {encode('Лев Толстой')}')
print('---')



# Encode the dataset and store it into torch.tensor
data = torch.tensor(encode(text), dtype=torch.long)



# Split dataset into train (90%) and validation (10%) sets
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]



# Split dataset on chunks
torch.manual_seed(1337)
block_size = 8                  # how many context characters we gonna use for prediction
batch_size = 4                  # how many independent sequences we gonna process in parallel

def get_batch(split: str) -> (list[int], list[int]):
    if split == 'train':
        data = train_data
    else:
        data = val_data

    ix = torch.randint(len(data) - block_size, (batch_size,))   # create 4 (batch_size) tensors of 8 (block_size) characters length on random positions in data
    x_list = []
    y_list = []
    for i in ix:
        x_list.append(data[i:i+block_size])
        y_list.append(data[i+1:i+block_size+1])
    return torch.stack(x_list), torch.stack(y_list)


xb, yb = get_batch('train')
print(f'Inputs: {xb.shape}')
print(xb)
print(f'Targets: {yb.shape}')
print(yb)

print('---')

for B in range(batch_size):
    print(f'Batch {B}')
    for T in range(block_size):
        context = xb[B, :T+1]
        target = yb[B, T]
        print(f'When input is {context.tolist()} the target is {target}')

print('---')



# The BigramLanguageModel is a simple neural network for language modeling (subclass of nn.Module)
# It predicts the next token in a sequence based only on the current token (bigram model).
class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size) -> None:
        super().__init__()
        # The embedding table maps each token to a vector of size vocab_size.
        # This allows the model to directly output logits for the next token.
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
    
    def forward(self, idx, targets):
        # idx: input tensor of shape (B, T), where B is batch size and T is sequence length.
        # targets: tensor of the same shape, containing the expected next tokens.
        # The embedding table returns logits for each position in the input.
        # Output shape: (B, T, vocab_size)
        logits = self.token_embedding_table(idx)
        B, T, C = logits.shape

        # Reshape logits and targets to be 2D tensors for cross-entropy loss calculation.
        # logits: (B, T, C) -> (B*T, C), targets: (B, T) -> (B*T)
        logits = logits.view(B*T, C)
        targets = targets.view(B*T)
        # Compute the cross-entropy loss between the predicted logits and the actual targets.
        loss = F.cross_entropy(logits, targets)

        return logits, loss

# Instantiate the model with the vocabulary size.
m = BigramLanguageModel(vocab_size)
logits, loss = m(xb, yb)

print(logits.shape)
print(loss)
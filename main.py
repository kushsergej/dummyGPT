import torch



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

for batch in range(batch_size):
    print(f'Batch {batch}')
    for ch in range(block_size):
        context = xb[batch, :ch+1]
        target = yb[batch, ch]
        print(f'When input is {context.tolist()} the target is {target}')
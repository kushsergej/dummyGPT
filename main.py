# Analyze dataset for vocab_size
with open('dataset.txt', 'r', encoding='utf-8') as file:
    text = file.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
print(f'Total chars in dataset is {vocab_size}')


# Implement encoder and decoder
stoi = {}
for i, ch in enumerate(chars):
    stoi[ch] = i
print(stoi)

itos = {}
for ch, i in enumerate(chars):
    itos[ch] = i
print(itos)
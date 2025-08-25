# Analyze dataset for vocab_size
from email import message


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
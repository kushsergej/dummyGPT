import torch

# read dataset
with open('dataset.txt', 'r', encoding='utf-8') as file:
    dataset = file.read()

print('Dataset length is ', len(dataset), ' characters')
# print('First 50 characters: ', dataset[:50])


# all unique characters in the dataset
chars = sorted(list(set(dataset)))
vocab_size = len(chars)
print('All possible characters in the dataset: ', ''.join(chars))
print('Vocabulary size is ', vocab_size, ' characters')


# create a mapping from characters to integers (tokenizer)
# https://github.com/openai/tiktoken.git
string_to_i = {}
for i, ch in enumerate(chars):
    string_to_i[ch] = i

i_to_string = {}
for i,ch in enumerate(chars):
    i_to_string[i] = ch

encode = lambda string: [string_to_i[char] for char in string]     # encode string to a list of integers
decode = lambda list: ''.join([i_to_string[i] for i in list])      # decode list of integers to string

print(decode(encode('Hello world')), ' --> ', encode('Hello world'),)


# Put dataset in torch
data = torch.tensor(data=encode(dataset), dtype=torch.long)
print(data.shape, data.type)
print(dataset[:50], ' --> ', data[:50])   # encode first 50 characters of dataset


# Split 90% of dataset as training data, and last 10% as validation data
splitter = int(0.9*len(data))
train_data = data[:splitter]
val_data = data[:splitter]
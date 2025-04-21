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
print('--- Tokenizer ---')
string_to_i = {}
for i, ch in enumerate(chars):
    string_to_i[ch] = i

i_to_string = {}
for i,ch in enumerate(chars):
    i_to_string[i] = ch

encode = lambda string: [string_to_i[char] for char in string]     # encode string to a list of integers
decode = lambda list: ''.join([i_to_string[i] for i in list])      # decode list of integers to string

print(decode(encode('Test')), ' --> ', encode('Test'),)


# Put dataset in torch
data = torch.tensor(data=encode(dataset), dtype=torch.long)
print(data.shape, data.type)
print(dataset[:5], ' --> ', data[:5])   # encode first 5 characters of dataset


# Split 90% of dataset as training data, and last 10% as validation data
splitter = int(0.9*len(data))
training_data = data[:splitter]
validation_data = data[:splitter]


# we do not train the LLM with a whole train_data, but with the chunks of it
print ('--- Chunks (example for 1 batch of data with chunk size 4) ---')
chunk_size = 4

context_set = training_data[:chunk_size]
print(f'context: {context_set.shape}')
print(context_set)

prediction_set = training_data[1: chunk_size+1]
print(f'prediction: {prediction_set.shape}')
print(prediction_set)

for position in range(chunk_size):
    context = context_set[:position+1]
    prediction = prediction_set[position]
    print(f'When context is {context.tolist()} the prediction: {prediction}')



print ('--- Chunks (for 4 batches of data with chunk size 10) ---')
torch.manual_seed(1337)
chunk_size = 10     # max context lengh for the prediction
batch_size = 4      # how many independent chunks will be processed in parallel

# generate a small batch of data of contexts and predictions
def get_batches(mode: str):
    if mode == 'train':
        data = training_data
    else:
        data = validation_data
    
    ix = torch.randint(len(data) - chunk_size, (batch_size,))
    context_set = torch.stack([data[i: i+chunk_size] for i in ix])
    prediction_set = torch.stack([data[i+1: i+chunk_size+1] for i in ix])
    return context_set, prediction_set


training_contexts, training_predictions = get_batches('train')
print(f'training contexts: {training_contexts.shape}')
print(training_contexts)
print(f'training predictions: {training_predictions.shape}')
print(training_predictions)



print('--- Proceed trainig data ---')
for batch in range(batch_size):
    for position in range(chunk_size):
        context = training_contexts[batch, :position+1]
        prediction = training_predictions[batch, position]
        print(f'When context is {context.tolist()} the prediction: {prediction}')
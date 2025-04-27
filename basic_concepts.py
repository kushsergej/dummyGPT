import torch
import torch.nn as nn
import torch.nn.functional as F


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

print(decode(encode('Лев Толстой')), ' --> ', encode('Лев Толстой'),)


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



# Enable Bigram LLM
torch.manual_seed(1337)

class BigramLLM(nn.Module):
    def __init__(self, vocab_size) -> None:
        super().__init__()
        # each token directly reads off the prediction for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, predictions=None):
        # idx and predictions are both (Batch,Time) tensor of integers
        llm_predictions = self.token_embedding_table(idx)    # (Batch,Time,Channel)
        if predictions is None:
            loss = None
        else:
            Batch, Time, Channel = llm_predictions.shape
            llm_predictions = llm_predictions.view(Batch*Time, Channel)
            predictions = predictions.view(Batch*Time)
            loss = F.cross_entropy(llm_predictions, predictions)
        return llm_predictions, loss

    def generate(self, idx, max_new_tokens):
        # idx is is (Batch,Time) array of indices in the current context
        for _ in range(max_new_tokens):
            # get predictions
            llm_predictions, loss = self(idx)
            # focus only on the last timestamp
            llm_predictions = llm_predictions[:, -1, :]     # becomes (Batch,Channel)
            # apply softmax to get probabilities
            probs = F.softmax(llm_predictions, dim=-1)      # (Batch,Channel)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (Batch, 1)
            # append sample index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)         # (Batch,Time+1)
        return idx


llm = BigramLLM(vocab_size)
llm_predictions, loss = llm(training_contexts, training_predictions)
print(llm_predictions.shape)
print(loss)

print('--- Work with BigramLLM (500 tokens, 1 optimization run) ---')
print(decode(
        llm.generate(
            idx = torch.zeros((1,1), dtype=torch.long),
            max_new_tokens=500
            )[0].tolist()
        )
    )


# create a PyTorch optimizer
print('--- optimize the loss ---')
print('--- Work with BigramLLM (500 tokens, 10.000 optimization runs) ---')
optimizer = torch.optim.AdamW(llm.parameters(), lr=0.001)

batch_size = 32
for steps in range(10000):
    # sample a batch of data
    training_contexts, training_predictions = get_batches('train')
    # evaluate the loss
    llm_predictions, loss = llm(training_contexts, training_predictions)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step


print(loss.item())
print(decode(
        llm.generate(
            idx = torch.zeros((1,1), dtype=torch.long),
            max_new_tokens=500
            )[0].tolist()
        )
    )
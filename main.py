import torch
import torch.nn as nn
import torch.nn.functional as F



# params
batch_size = 32     # how many independent sequences would we process in parallel
block_size = 8      # maximum content length for prediction
max_iters = 3000
eval_interval = 300
learning_rate = 0.01
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
number_of_embedded_dimensions = 32
# ---



torch.manual_seed(1337)



# read dataset
with open('dataset.txt', 'r', encoding='utf-8') as file:
    dataset = file.read()



# here is all unique characters that occurs in dataset
chars = sorted(list(set(dataset)))
vocab_size = len(chars)
print('All possible characters in the dataset: ', ''.join(chars))
print('Vocabulary size is ', vocab_size, ' characters')



# create a mapping from characters to integers (tokenizer)
# https://github.com/openai/tiktoken.git
print('--- Tokenizer ---')
string_to_tokens = {}
for i, ch in enumerate(chars):
    string_to_tokens[ch] = i

tokens_to_string = {}
for i,ch in enumerate(chars):
    tokens_to_string[i] = ch

encode = lambda string: [string_to_tokens[c] for c in string]           # encode string to a list of integers
decode = lambda list: ''.join([tokens_to_string[i] for i in list])      # decode list of integers to string

print(decode(encode('Лев Толстой')), ' --> ', encode('Лев Толстой'))



# Split 90% of dataset as training data, and last 10% as validation data
data = torch.tensor(encode(dataset), dtype=torch.long)
splitter = int(0.9*len(data))
training_data = data[:splitter]
validation_data = data[:splitter]



# data loading
def get_batches(mode: str):
    data = training_data if mode == 'train' else validation_data
    data_slices = torch.randint(len(data) - block_size, (batch_size,))

    context_set = torch.stack([data[i: i + block_size] for i in data_slices])
    prediction_set = torch.stack([data[i + 1: i + block_size + 1] for i in data_slices])
    context_set, prediction_set = context_set.to(device), prediction_set.to(device)

    return context_set, prediction_set



# Enable Bigram LLM
class BigramLLM(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        # each token directly reads off the prediction for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, number_of_embedded_dimensions)
        self.position_embedding_table = nn.Embedding(vocab_size, number_of_embedded_dimensions)
        self.llm_head = nn.Linear(number_of_embedded_dimensions, vocab_size)

    def forward(self, data_slice, predictions=None):
        Batch, Time = data_slice.shape
        # data_slice and predictions are both (Batch,Time) tensor of integers
        token_embedding = self.token_embedding_table(data_slice)                                    # (Batch,Time,Channel)
        position_embedding = self.position_embedding_table(torch.arange(Time, device = device))     # (Time,Channel)
        x = token_embedding + position_embedding                                                    # (Batch,Time,Channel)
        llm_predictions = self.llm_head(x)                                                          # (Batch,Time,vocab_size)
        if predictions is None:
            loss = None
        else:
            Batch, Time, Channel = llm_predictions.shape
            llm_predictions = llm_predictions.view(Batch*Time, Channel)
            predictions = predictions.view(Batch*Time)
            loss = F.cross_entropy(llm_predictions, predictions)
        return llm_predictions, loss

    def generate(self, data_slice, max_new_tokens):
        # data_slice is (Batch,Time) array of indices in the current context
        for _ in range(max_new_tokens):
            # get predictions
            llm_predictions, loss = self(data_slice)
            # focus only on the last timestamp
            llm_predictions = llm_predictions[:, -1, :]                     # becomes (Batch,Channel)
            # apply softmax to get probabilities
            probs = F.softmax(llm_predictions, dim = -1)                    # (Batch,Channel)
            # sample from the distribution
            data_slice_next = torch.multinomial(probs, num_samples = 1)     # (Batch, 1)
            # append sample index to the running sequence
            data_slice = torch.cat((data_slice, data_slice_next), dim=1)    # (Batch,Time+1)
        return data_slice


llm = BigramLLM()
model = llm.to(device)



@torch.no_grad()
def estimate_loss():
    out = {}
    llm.eval()

    for mode in ('train', 'validate'):
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            contexts, predictions = get_batches(mode)
            llm_prediction, loss = llm(contexts, predictions)
            losses[k] = loss.item()
        out[mode] = losses.mean()

    llm.train()
    return out



# create a PyTorch optimizer
optimizer = torch.optim.AdamW(llm.parameters(), lr = learning_rate)

# create trainig loop
for iter in range(max_iters):
    # every once in a while, evaluate the loss on training and validation sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f'Step {iter}: training loss {losses['train']:.4f}, validation loss {losses['validate']:.4f}')
    
    # sample a batch of data
    contexts_per_batch, predictions_per_batch = get_batches('train')

    # evaluate the loss
    llm_predictions, loss = llm(contexts_per_batch, predictions_per_batch)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()



# Generate from the LLM
context = torch.zeros((1, 1), dtype=torch.long, device = device)
print(decode(model.generate(context, max_new_tokens = 500)[0].tolist()))
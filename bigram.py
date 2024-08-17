# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparams
batch_size = 32
block_size = 8
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 32

# ----------
torch.manual_seed(1337)

# We always start with a dataset to train on. Let's download the tiny shakespeare dataset
# !wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

# read to inspect
with open('input.txt', 'r', encoding = 'utf-8') as f:
  text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)

# create a dictionary mapping from characters to integers
stoi = {ch:i for i,ch in enumerate(chars)} #enumerate allows you to basically iterate over several things without needing a manual counter
itos = {i:ch for i,ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s] # take a string, output list of ints
decode = lambda l: ''.join([itos[i] for i in l]) #take a list of ints, output a string

## char level tokenizer, most people have subword tokenizers, 50k tokens of varying lengths/chunks

# we use torch / tensors because they are more efficient and structure to perform
#operations and transformations and products as opposed to multidim arrays
data = torch.tensor(encode(text), dtype = torch.long)

#split up data into train and val sets
n = int(0.9 * len(data)) #first 90% of chars will be train, rest val
train_data = data[:n]
val_data = data[n:]


# mini batches of multiple chunks of text, stacked up in a single tensor
# done for efficiency to keep the GPU busy, they dont talk to each other

def get_batch(split):
  #generate a small batch of data of inputs x and targets y
  data = train_data if split == 'train' else val_data
  # generate batch size number different indices to start sampling data
  # in this case, ix will be a tensor containing 4 elements
  # batch_size being passed as a single element tuple to specify output shape
  ix = torch.randint(len(data) - block_size, (batch_size,))

  # create two different tensor stacks/batches, one for inputs and one for targets
  x = torch.stack([data[i:i+block_size] for i in ix])
  y = torch.stack([data[i+1:i+block_size+1] for i in ix])
  x, y = x.to(device), y.to(device)
  return x, y

# tell pytorch we dont need to do backprop
@torch.no_grad()
def estimate_loss(): # averages out the loss over multiple batches
  out = {} #returns a dict with split: loss for train and val
  model.eval() #set to evaluation phase
  for split in ['train', 'val']:
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
      X, Y = get_batch(split)
      logits, loss = model(X, Y)
      losses[k] = loss.item()
    out[split] = losses.mean() #get average loss for both splits
  model.train() #set to training phase
  return out

class BigramLanguageModel(nn.Module):

  def __init__(self):
    super().__init__()
    # each token directly reads off the logits for the next token from a lookup table
    # logits being the unnormalized probability or raw score that a token is next
    # declare look up table to have number of embeddings = number of dims = vocab size
    # embeddings are just when input tokens are converted into vectors
    # nn.Embedding is like a wrapper for a tensor of shape ( , ), initializes to some values?
    self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
    self.positional_embedding_Table = nn.Embedding(block_size, n_embd)
    self.lm_head = nn.Linear(n_embd, vocab_size) # input size, output size

  def forward(self, idx, targets=None):
    B,T = idx.shape
    # idx and targets are both (B,T) (batch, time) tensor of integers
    # for each index, pluck out the row of the embedding table containing logits,
    # which has vocab_size number of channels
    tok_emb = self.token_embedding_table(idx) # (Batch = 4, Time = 8 (block size?), Channels = 65)
    pos_emb = self.positional_embedding_Table(torch.arange(T, device=device)) #(T,C)
    x = tok_emb + pos_emb
    logits = self.lm_head(x) # (B,T, vocab_size)

    if targets is None:
      loss = None
    else:
      # must reshape logits due to pytorch implementation of cross entropy
      B, T, C = logits.shape
      logits = logits.view(B*T, C) #2D
      targets = targets.view(B*T) #1D, view is more efficient over reshape for contiguous tensors

      # cross entropy loss measures diff between the predicted probability distribution and true probability distribution of the labels
      # penalizes wrong predictions heavily, smooth and differentiable, suited for multi-class classification
      # softmax converts logits (raw scores) into a probability distribution over classes, which is then used to compute cross entropy loss
      # for multi class: loss = - summation across # of classes (true label) log (predicted probability)
      # so this is just for one single character, for each batch
      loss = F.cross_entropy(logits, targets)

    return logits, loss

  # the idea is to keep concatenating chars to T for max_new_tokens chars
  def generate(self, idx, max_new_tokens):
    # idx is (B,T) array of indices in the current context
    for _ in range(max_new_tokens):
      # get the predictions
      # no targets are provided, so loss is none, we don't need to look at loss while generating shit
      logits, loss = self(idx) # omg this calls the forward method of this class ohh

      # focus only on the last time step
      # for each batch, there is the vocab size distribution for next possible char
      logits = logits[:, -1, :] #becomes(B,C), the -1 basically subtracts the dimension in the middle

      # apply softmax to get probabilities over each batch of vocab_size
      # note: vocabulary includes both start token and end token probably
      probs = F.softmax(logits, dim =-1) # (B, C)

      # sample the next char (tensor of chars bc batch) from the distribution
      # torch.multinomial returns a tensor where each row contains num_samples indices sampled from the multinomial
      # in this case, num_samples = 1
      # samples probabilistically, so higher weights/probs will be more likely to get selected
      idx_next = torch.multinomial(probs, num_samples=1) # (B,1)

      #append sampled index to the running sequence
      idx = torch.cat((idx, idx_next), dim = 1) # (B, T+1)
    return idx

model = BigramLanguageModel()
m = model.to(device)

# create a PyTorch optimizer, AdamW much more popular and advanced than SGD
# an optimizer is an implementation that is used to update parameters of a neural network
optimizer = torch.optim.AdamW(m.parameters(), lr = learning_rate) #higher learning rate for small samples

# training loop

for iter in range(max_iters):

  # every once in an eval_interval, evaluate loss on train and val sets
  if iter % eval_interval ==0:
    losses = estimate_loss()
    print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:4f}")
  
  xb, yb = get_batch('train')

  # evaluate the loss
  logits, loss = model(xb, yb)
  optimizer.zero_grad(set_to_none = True) #so gradients dont accumulate across iters
  loss.backward() #backprop and compute gradients
  optimizer.step()

#generate from the model
context = torch.zeros((1,1), dtype = torch.long, device=device)
print(decode(m.generate(context, max_new_tokens = 500)[0].tolist()))

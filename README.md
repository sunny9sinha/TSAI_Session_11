# TSAI_Session_11

As part of the assignment we need to submit the readme file that must explain Encoder/Decoder Feed-forward manual steps and the attention mechanism that we have used.

Encoder Steps:
Considering our input language as French and target language as English. We define our input size for embedding layer as total number of words in French language text we have, that is 4345. And we define the hidden size that is the size of embedding for each word as 256. Then we define our Embedding layer with parameters input size and hidden size.

```python
input_size = input_lang.n_words

hidden_size = 256

embedding = nn.Embedding(input_size, hidden_size).to(device)
````

As our assignment requires us to replace GRU in class code with LSTM, so we define the LSTM layer as :

```python
lstm = nn.LSTM(hidden_size, 
                           hidden_size, 
                           num_layers=1, 
                           dropout=0.2,
                           batch_first=True).to(device)
 ````

Taking a random sample as below :
```python
sample = random.choice(pairs)
sample

['ils sont russes .', 'they are russian .']
````
Taking these sentences as input and target:
```python
input_sentence = sample[0]
target_sentence = sample[1]
````
Using list comprehension and creating the input sentence, target sentence to input indices and target indices. Then adding EOS token at the end of indices list.
```python
input_indices = [input_lang.word2index[word] for word in input_sentence.split(' ')]
target_indices = [output_lang.word2index[word] for word in target_sentence.split(' ')]
input_indices, target_indices

([348, 349, 1459, 5], [221, 124, 804, 4])

input_indices.append(EOS_token)
target_indices.append(EOS_token)
input_indices, target_indices

([348, 349, 1459, 5, 1], [221, 124, 804, 4, 1])
````
Now we convert these list of indices to tensors as our embedding layer expects the input in tensors:
```python
input_tensor = torch.tensor(input_indices, dtype=torch.long, device= device)
output_tensor = torch.tensor(target_indices, dtype=torch.long, device= device)
````

Now we fake the batch size to embedding layer using the view method and pass the input tensor to embedding layer, then the embedded input is passed to the LSTM layer  as following:
```python
embedded_input = embedding(input_tensor[0].view(-1, 1))
output, (hidden, cell) = lstm(embedded_input)
````
Combining all the code above and running it for complete setence in a loop like this:
```python
encoder_outputs = torch.zeros(MAX_LENGTH, 256, device=device)
encoder_hidden = torch.zeros(1, 1, 256, device=device)

for i in range(input_tensor.size()[0]):
  embedded_input = embedding(input_tensor[i].view(-1, 1))
  output, (encoder_hidden, cell) = lstm(embedded_input)
  encoder_outputs[i] += output[0,0]
````

Decoder Steps with Attention:

We take decoder first input as a tensor with SOS token, and decoder hidden state and cell state is taken from encoder LSTM only. 
```python
decoder_input = torch.tensor([[SOS_token]], device=device)
decoder_hidden = encoder_hidden
decoder_cell = cell
decoded_words = []
````

```python
output_size = output_lang.n_words

embedding = nn.Embedding(output_size, 256).to(device)
````

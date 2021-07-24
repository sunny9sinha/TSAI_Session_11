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
Output size is same as the number of words in the output language, we define the decoder embedding layer as follows.
```python
output_size = output_lang.n_words

embedding = nn.Embedding(output_size, 256).to(device)

embedded = embedding(decoder_input)
embedded.shape
````
We are using the attention mechanism as discussed in class which is the combination of Bahdanau Attention and Loung Attention as expained on Pytorch tutorial:
```python
import torch.nn.functional as F

attn_weight_layer = nn.Linear(256 * 2, 10).to(device)
attn_weights = attn_weight_layer(torch.cat((embedded[0], decoder_hidden[0]), 1))
attn_weights = F.softmax(attn_weights, dim = 1)
attn_weights
````
Then we perform the batch matrix multiplication of attention weights and encoder outputs.
```python
attn_applied = torch.bmm(attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(0))
attn_applied.shape

torch.Size([1, 1, 256])
````
Then we define the input for LSTM layer by concatenating the embedded tensor and attention applied tensor. We use unsqueeze to fake the batch size.
```pyhton
input_to_lstm = input_to_lstm_layer(torch.cat((embedded[0], attn_applied[0]), 1))
input_to_lstm.shape

torch.Size([1, 256])

input_to_lstm = input_to_lstm.unsqueeze(0).to(device)
decoder_hidden.shape, input_to_lstm.shape


(torch.Size([1, 1, 256]), torch.Size([1, 1, 256]))
````

We then define LSTM layer and pass the parameters accordingly:
```python
lstm = nn.LSTM(256, 256,batch_first = True).to(device)
output, (decoder_hidden,decoder_cell) = lstm(input_to_lstm, (decoder_hidden,decoder_cell))
output.shape, decoder_hidden.shape

(torch.Size([1, 1, 256]), torch.Size([1, 1, 256]))
````

Then we define the linear layer that would give us the output word:
```python
output_word_layer = nn.Linear(256, output_lang.n_words).to(device)
````

Combining the code above and passing it for the complete sample sentence in a loop, also taking the relu on the lstm output and softmax on the linear layer output as required we get:

```pyhton
for i in range(4):
  decoder_input = torch.tensor([[target_indices[i]]], device=device)
  decoder_hidden = encoder_hidden
  decoder_cell = cell
  output_size = output_lang.n_words
  embedded = embedding(decoder_input)
  attn_weights = attn_weight_layer(torch.cat((embedded[0], decoder_hidden[0]), 1))
  attn_weights = F.softmax(attn_weights, dim = 1)
  attn_applied = torch.bmm(attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(0))
  input_to_lstm = input_to_lstm_layer(torch.cat((embedded[0], attn_applied[0]), 1))
  input_to_lstm = input_to_lstm.unsqueeze(0)
  output, (decoder_hidden,decoder_cell) = lstm(input_to_lstm, (decoder_hidden,decoder_cell))
  output = F.relu(output)
  output = F.softmax(output_word_layer(output[0]), dim = 1)
  top_value, top_index = output.data.topk(1)
  print(target_sentence.split(" ")[i], target_indices[i], output_lang.index2word[top_index.item()], top_index.item() )
  print(attn_weights)
  ````

We get output as:
```python
they 221 hearted 1785
tensor([[0.0759, 0.1077, 0.1252, 0.1332, 0.0657, 0.1513, 0.0645, 0.1016, 0.1151,
         0.0598]], device='cuda:0', grad_fn=<SoftmaxBackward>)
are 124 hearted 1785
tensor([[0.0471, 0.1107, 0.1110, 0.1555, 0.0926, 0.1097, 0.0611, 0.1249, 0.1290,
         0.0585]], device='cuda:0', grad_fn=<SoftmaxBackward>)
russian 804 stylish 1281
tensor([[0.1193, 0.0479, 0.1065, 0.1028, 0.1047, 0.0778, 0.1301, 0.1245, 0.0601,
         0.1261]], device='cuda:0', grad_fn=<SoftmaxBackward>)
. 4 stylish 1281
tensor([[0.1252, 0.0939, 0.0971, 0.1561, 0.1925, 0.0892, 0.0782, 0.0321, 0.0601,
         0.0758]], device='cuda:0', grad_fn=<SoftmaxBackward>)
````



import torch,math
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class LSTMEncoder(nn.Module):
	def __init__(self, input_size, embedding_size, hidden_size):
		"""
		Inputs: input, (h_0, c_0)
		Outputs: output, (h_n, c_n)
			- |input| = features of the input sequence.
				  = (seq_len, batch_size, input_size)
			- |h_0| = initial hidden state for each element in the batch.
				= (num_layers * num_directions, batch_size, hidden_size)
			- |c_0| = initial cell state for each element in the batch.
				= (num_layers * num_directions, batch_size, hidden_size)
			- |output| = output features (h_t) from the last layer of the LSTM, for each t.
				   = (seq_len, batch_size, num_directions * hidden_size)
			- |h_n| = hidden state for t = seq_len
				= (num_layers * num_directions, batch_size, hidden_size)
			- |c_n| = cell state for t = seq_len.
				= (num_layers * num_directions, batch_size, hidden_size)
		"""
		super(LSTMEncoder, self).__init__()

		self.input_size = input_size
		self.embedding_size = embedding_size
		self.hidden_size = hidden_size
		self.embedding_pretrained = torch.tensor(np.load('./embedding_zi.npy').astype('float32'))
		# layers
		self.embedding = nn.Embedding.from_pretrained(self.embedding_pretrained, freeze=False)
		self.lstm = nn.LSTM(embedding_size, hidden_size, bidirectional = False, batch_first = True)

	def forward(self, input, hidden):
		
		out = self.embedding(input).view(1, 1, -1)
	
		output, hidden = self.lstm(out, hidden)
		
		return output, hidden

	def initHidden(self):

		return torch.zeros(1, 1, self.hidden_size)
	
class Attention(nn.Module):
	def __init__(self, hidden_size):
		super(Attention, self).__init__()
		self.hidden_size = hidden_size
		self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
		self.v = nn.Parameter(torch.rand(hidden_size))
		stdv = 1. / math.sqrt(self.v.size(0))
		self.v.data.uniform_(-stdv, stdv)

	def forward(self, hidden, encoder_outputs):
		timestep = encoder_outputs.size(0)
		h = hidden.repeat(timestep, 1, 1).transpose(0, 1)
		encoder_outputs = encoder_outputs.transpose(0, 1)  # [B*T*H]
		attn_energies = self.score(h, encoder_outputs)
		return F.softmax(attn_energies, dim=1).unsqueeze(1)

	def score(self, hidden, encoder_outputs):
		# [B*T*2H]->[B*T*H]
		energy = F.relu(self.attn(torch.cat([hidden, encoder_outputs], 2)))
		energy = energy.transpose(1, 2)  # [B*H*T]
		v = self.v.repeat(encoder_outputs.size(0), 1).unsqueeze(1)  # [B*1*H]
		energy = torch.bmm(v, energy)  # [B*1*T]
		return energy.squeeze(1)  # [B*T]

class LSTMDecoder(nn.Module):
	def __init__(self, output_size, embedding_size, hidden_size):
		super(LSTMDecoder, self).__init__()

		self.output_size = output_size
		self.embedding_size = embedding_size
		self.hidden_size = hidden_size
		self.attention = Attention(hidden_size)
		# layers
		self.embedding_pretrained = torch.tensor(np.load('./embedding_zi.npy').astype('float32'))
		# layers
		self.embedding = nn.Embedding.from_pretrained(self.embedding_pretrained, freeze=False)
		self.lstm = nn.LSTM(embedding_size+hidden_size, hidden_size,  batch_first = True)

		self.out = nn.Linear(hidden_size, output_size)
		self.softmax = nn.LogSoftmax(dim = 1)


	def forward(self, input, hidden, encoder_outputs):
		# Calculate attention weights and apply to encoder outputs
		attn_weights = self.attention(hidden[-1], encoder_outputs)
		context = attn_weights.bmm(encoder_outputs.transpose(0, 1))  # (B,1,N)
		context = context.transpose(0, 1)  # (1,B,N)
	
		output = self.embedding(input).view(1, 1, -1)
		output = F.relu(output)
		lstm_input = torch.cat([output, context], 2)
		# |output| = (1, 1, hidden_size)

		output, hidden = self.lstm(lstm_input, hidden)

		output = self.softmax(self.out(output[0]))
		# |output| = (1, output_lang.n_words)
		return output, hidden

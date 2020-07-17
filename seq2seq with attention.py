import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator
import numpy as np
import spacy
import random

spacy_ger = spacy.load('de')
spacy_eng = spacy.load('en')


def tokenizer_ger(text):
    return [tok.text for tok in spacy_ger.tokenizer(text)]


# "Hello my name is Rishabh " -> ["Hello", "my", "name", "is", "Rishabh"]

def tokenizer_eng(text):
    return [tok.text for tok in spacy_eng.tokenizer(text)]


# For pre processing we define field

german = Field(tokenize=tokenizer_ger, lower=True,
               init_token='<sos>', eos_token='eos')

english = Field(tokenize=tokenizer_eng, lower=True,
                init_token='<sos>', eos_token='eos')

train_data, validation_data, test_data = Multi30k.splits(exts=('.de', '.en'), fields=(german, english))

german.build_vocab(train_data, max_size=10000, min_freq=2)
# if a word is repeated only once we will not add to vocab
english.build_vocab(train_data, max_size=10000, min_freq=2)


#       input size = size of the german vocabulary
#       embedding size = To map to n dimensional


class Encoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers, p):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.dropout = nn.Dropout(p)
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=p)

    def forward(self, x):
        # shape of x = (seq_length , batch size)
        embedding = self.dropout(self.embedding(x))
        # shape of embedding = (seq_length , batch size, embedding size)
        output, (hidden, cell) = self.rnn(embedding)

        return hidden, cell


class Decoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, output_size, num_layers, p):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.dropout = nn.Dropout(p)
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=p)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden, cell):
        # shape of x =  [N] but we want [1, N] for doing one word at a time
        x.unsqueeze(0)  # this will add one dim
        embedding = self.dropout(self.embedding(x))
        # shape of embedding = (1, N, embedding size)
        outputs, (hidden, cell) = self.rnn(embedding, (hidden, cell))
        predictions = self.fc(outputs)
        predictions = predictions.squeeze(0)
        return predictions, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, source, target, teacher_force_ratio=0.5):
        batch_size = source.shape[1]
        target_len = target.shape[0]
        target_vocab_size = len(english.vocab)

        outputs = torch.zeros(target_len, batch_size, target_vocab_size).to(device)

        hidden, cell = self.encoder(source)

        # grab start token
        x = target[0]

        for t in range(1, target_len):
            outputs, hidden, cell = self.decoder(x, hidden, cell)

            outputs[t] = outputs
            # (N, Vocab_size)
            best_guess = outputs.agrmax(1)

            x = target[t] if random.random() < teacher_force_ratio else best_guess

        return outputs


# Training Parameters :
num_of_epoch = 20
learning_rate = 0.001
batch_size = 64

# Model Parameters :
load_model = False
device = torch.device('cude' if torch.cuda.is_available() else 'cpu')
input_size_encoder = len(german.vocab)
input_size_decoder = len(english.vocab)
output_size = len(english.vocab)
encoder_embedding_size = 300
decoder_embedding_size = 300
hidden_size = 1024
num_layers = 2
enc_dropout = 0.5
dec_dropout = 0.5

train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_size=batch_size,
    sort_within_batch=True,
    sort_key=lambda x: len(x.src),
    device=device,
)

encoder_net = Encoder(
    input_size_encoder, encoder_embedding_size, hidden_size, num_layers, enc_dropout
).to(device)

decoder_net = Decoder(
    input_size_decoder,
    decoder_embedding_size,
    hidden_size,
    output_size,
    num_layers,
    dec_dropout,
).to(device)

model = Seq2Seq(encoder_net, decoder_net).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

pad_idx = english.vocab.stoi["<pad>"]
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)


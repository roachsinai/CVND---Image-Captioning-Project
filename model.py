import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import copy
from heapq import heappush, heappop

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)

        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.bn = nn.BatchNorm1d(resnet.fc.in_features)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images) # shape: batch_size, 2048, 1, 1
        features = features.view(features.size(0), -1) # shape: batch_size, 2048
        features = self.bn(features)
        features = self.embed(features)
        return features


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super().__init__()

        self.hidden_size = hidden_size
        self.word_embeddings = nn.Embedding(vocab_size, embed_size)

        self.lstm = nn.LSTM(input_size=embed_size,
                            hidden_size=hidden_size,
                            num_layers=1,
                            bias=True,
                            batch_first=True,
                            dropout=0,
                            bidirectional=False)

        self.linear = nn.Linear(hidden_size, vocab_size)

    def init_hidden(self, batch_size):
        """shape: (num_layers, batch_size, hidden_dim)"""
        return (torch.zeros((1, batch_size, self.hidden_size), device=device),
                torch.zeros((1, batch_size, self.hidden_size), device=device))

    def forward(self, features, captions):
        """features.shape: (batch_size, embed_size)"""

        batch_size = features.shape[0]
        self.hidden = self.init_hidden(batch_size)

        # 去掉每个caption的<end>标识
        captions = captions[:, :-1]
        embeddings = self.word_embeddings(captions)
        # embeddings shape:(batch_size, caption length, embed_size)，为了和embeddings拼接，unsqueeze features
        embeddings = torch.cat((features.unsqueeze(1), embeddings), dim=1)

        # Get the output and hidden state
        lstm_out, self.hidden = self.lstm(embeddings, self.hidden)

        # Fully connected layer, (batch_size, caption length, vocab_size)
        outputs = self.linear(lstm_out)

        return outputs

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        output = []
        # batch_size is 1 at inference
        batch_size = inputs.shape[0] # (1, 1, embed_size)
        hidden = self.init_hidden(batch_size)

        while True:
            lstm_out, hidden = self.lstm(inputs, hidden)
            outputs = self.linear(lstm_out) # (1, 1, vocab_size)
            outputs = outputs.squeeze(1)
            _, max_indice = torch.max(outputs, dim=1)

            output.append(max_indice.cpu().numpy()[0].item())

            # if we get <end> word, break
            if (max_indice == 1):
                break

            inputs = self.word_embeddings(max_indice)
            inputs = inputs.unsqueeze(1)

        return output

    def beam_search_sample(self, inputs, beam=3):
        # batch_size is 1 at reference
        batch_size = inputs.shape[0]
        hidden = self.init_hidden(batch_size)

        # sequences[0][0] : log probability of the word seq predicted
        # sequences[0][1] : index of word of this seq, initialize a Tensor([0]) for programming convenient
        # sequences[0][2] : hidden state related of the last word
        # sequences[0][3] : count of words in this seq except start and end word
        sequences = [[.0, [torch.Tensor([0]).long().to(device)], hidden, 0]]
        max_len = 20
        sorted_seqs = []

        # Step 1
        # Predict the first word <start>
        outputs, hidden = DecoderRNN.get_outputs(self, inputs, hidden)
        softmax_score = F.log_softmax(outputs, dim=1) # Define a function to sort the cumulative socre
        sorted_score, indices = torch.sort(-softmax_score, dim=1)
        word_preds = indices[0][:beam]
        best_scores = sorted_score[0][:beam]
        _, max_indice = torch.max(outputs, dim=1)

        for i in range(beam):
            seq = copy.deepcopy(sequences[0])
            seq[1].append(word_preds[i])
            if word_preds[i] != 1:
                seq[3] += 1
            seq[0] = (seq[0] + best_scores[i]) / seq[3]
            seq[2] = hidden
            heappush(sorted_seqs, seq)
        sequences = sorted_seqs[:beam]

        # self.echo_sequences(sequences, beam)

        sorted_seqs=[]

        l = 1
        while l < max_len:
            # print("l:", l)
            l += 1
            temp = []
            for seq in sequences:
                inputs = seq[1][-1] # last word index in seqences
                if inputs == 1:
                    heappush(sorted_seqs, seq)
                    continue
                inputs = inputs.type(torch.cuda.LongTensor)
                # Embed the input word
                inputs = self.word_embeddings(inputs) # inputs shape : (1, embed_size)
                inputs = inputs.reshape((1, 1, -1)) # inputs shape : (1, 1, embed_size)

                # retrieve the hidden state
                hidden = seq[2]
                preds, hidden = DecoderRNN.get_outputs(self, inputs, hidden)

                # Getting the top <beam_index>(n) predictions
                softmax_score = F.log_softmax(preds, dim=1) # Define a function to sort the cumulative socre
                sorted_score, indices = torch.sort(-softmax_score, dim=1)
                word_preds = indices[0][:beam]
                best_scores = sorted_score[0][:beam]
                for i in range(beam):
                    sub_seq = seq.copy()
                    sub_seq[1] = seq[1][:]
                    sub_seq[1].append(word_preds[i])
                    if word_preds[i] != 1:
                        sub_seq[3] += 1
                    sub_seq[0] = (sub_seq[0] + best_scores[i]) / sub_seq[3]
                    sub_seq[2] = hidden
                    heappush(sorted_seqs, sub_seq)

            sequences = sorted_seqs[:beam]
            sorted_seqs = []
            # self.echo_sequences(sequences, beam)

        for i in range(beam):
            sequences[i] = sequences[i][1][1:] # remove the first initialized Tensor([0])
            for j, elem in enumerate(sequences[i]):
                sequences[i][j] = elem.cpu().numpy().item()
            if sequences[i][-1] != 1:
                sequences[i].append(1)

        return sequences

    def get_outputs(self, inputs, hidden):
        lstm_out, hidden = self.lstm(inputs, hidden) # lstm_out shape: (1, 1, hidden_size)
        outputs = self.linear(lstm_out) # outputs shape: (1, 1, vocab_size)
        outputs = outputs.squeeze(1)

        return outputs, hidden

    def get_next_word_input(self, max_indice):
        # Prepare to embed the last predicted word to be the new input of the lstm
        inputs = self.word_embeddings(max_indice)
        inputs = inputs.unsqueeze(1)

        return inputs

    def echo_sequences(self, sequences, beam):
        for i in range(beam):
            p_seq = []
            print(sequences[i][1])
            for j in sequences[i][1]:
                p_seq.append(j.cpu().numpy().item())
            print(p_seq)

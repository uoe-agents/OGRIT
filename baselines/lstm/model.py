import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    def __init__(self, in_shape, lstm_hidden_shape, fc_hidden_shape, out_shape, num_layers=2, bias=True, dropout=0.0):
        super(LSTMModel, self).__init__()

        self.lstm = nn.LSTM(in_shape, lstm_hidden_shape, num_layers=num_layers, bias=bias,
                            dropout=dropout, batch_first=True)
        self.fc = nn.Sequential(nn.Linear(lstm_hidden_shape, fc_hidden_shape),
                                nn.ReLU(),
                                nn.Dropout(dropout),
                                nn.Linear(fc_hidden_shape, out_shape),
                                nn.Dropout(dropout),
                                nn.LogSoftmax(dim=-1))

        for layer in self.fc:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.constant_(layer.bias, 0.0)

    def forward(self, x, use_encoding=False):
        # todo: n_n = hidden state | c_n = cell stateF

        encoding, (h_n, c_n) = self.lstm(x)
        output = self.fc(h_n[-1])
        if use_encoding:
            encoding, lengths = nn.utils.rnn.pad_packed_sequence(encoding, batch_first=True)
            mask = (torch.arange(encoding.shape[1])[None, :] >= lengths[:, None]).to(encoding.device)
            encoding = encoding.masked_fill(mask.unsqueeze(-1).repeat(1, 1, encoding.shape[-1]), 0.0)
            encoding = self.fc(encoding)
            return output, (encoding, lengths)
        else:
            return output, (None, None)

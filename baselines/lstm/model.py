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
        # todo: H_n = hidden state | c_n = cell stateF | encoding = output of lstm ("last" depth-wise layer)

        encoding, (h_n, c_n) = self.lstm(x)
        output = self.fc(h_n[-1])

        # todo: if we want to compute the goal probabilities for the trajectories at each time step
        if use_encoding:
            encoding, lengths = nn.utils.rnn.pad_packed_sequence(encoding, batch_first=True, padding_value=0.0)

            # todo: we pass through fc as we do for the output above to get probabilities for the goals
            encoding = self.fc(encoding)
            return output, (encoding, lengths)
        else:
            return output, (None, None)

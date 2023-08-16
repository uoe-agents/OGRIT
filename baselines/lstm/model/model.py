import torch
import torch.nn as nn

from ogrit.core.base import LSTM_PADDING_VALUE


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
                                nn.Sigmoid())
        self.float()

        for layer in self.fc:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.constant_(layer.bias, 0.0)

    def forward(self, trajectory, lengths: torch.Tensor, use_encoding=True, device='cuda'):

        x = trajectory.to(device)

        # h_n = hidden state | c_n = cell state | encoding = output of lstm ("last" depth-wise layer)
        encoding, (h_n, c_n) = self.lstm(x)
        final_prediction = self.fc(h_n[-1]).to(device)

        # If we want to compute the goal probabilities for the trajectories at each time step use_encoding=True,
        # otherwise use_encoding=False to only get the prediction at the last time step of the trajectory.
        if use_encoding:

            mask = (torch.arange(encoding.shape[1])[None, :].to(device) >= lengths[:, None]).to(device)
            encoding = encoding.masked_fill(mask.unsqueeze(-1).repeat(1, 1, encoding.shape[-1]), LSTM_PADDING_VALUE)

            # Use the linear layer to get the goal probabilities at each time step
            intermediate_predictions = self.fc(encoding).to(device)
            return final_prediction, intermediate_predictions
        else:
            return final_prediction, None

    def predict_zeros(self):
        self._predict_zeros = True

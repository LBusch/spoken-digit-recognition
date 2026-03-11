import torch
import torch.nn as nn

class ASRModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, gru_layers):
        super().__init__()
        # convolutional layers for feature extraction
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, padding=1),
            nn.ReLU()
        )
        # recurrent layers for temporal modeling
        self.gru = nn.GRU(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=gru_layers, batch_first=True, bidirectional=True)
        self.pooling = nn.AdaptiveAvgPool1d(1)
        # fully connected layer for classification output
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x, feature_lengths):
        # x: (batch, time, mel)
        x = x.transpose(1, 2)   # (batch, mel, time)
        x = self.conv(x)
        x = x.transpose(1, 2)   # (batch, time, hidden_dim)
        x = nn.utils.rnn.pack_padded_sequence(x, lengths=feature_lengths, batch_first=True, enforce_sorted=False)
        x, _ = self.gru(x)      # (batch, time, hidden_dim * 2)
        x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        x = x.transpose(1, 2)   # (batch, hidden_dim * 2, time)
        x = self.pooling(x)
        x = x.squeeze(-1)       # (batch, hidden_dim * 2)
        x = self.fc(x)          # (batch, output_dim)  
        return x
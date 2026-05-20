import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionLayer(nn.Module):
    def __init__(self, hidden_dim):
        super(AttentionLayer, self).__init__()
        # In BiLSTM, hidden_dim is usually doubled (forward + backward)
        self.W_h = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.v_a = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, lstm_out):
        # lstm_out shape: (batch_size, time_steps, hidden_dim)
        
        # Calculate attention scores: S_t = v_a^T * tanh(W_h * h_t + b_h)
        # 1. Apply linear transformation and tanh
        x = torch.tanh(self.W_h(lstm_out))  # (batch_size, time_steps, hidden_dim)
        
        # 2. Multiply by v_a
        scores = self.v_a(x).squeeze(-1)    # (batch_size, time_steps)
        
        # 3. Softmax to get weights
        alpha = F.softmax(scores, dim=1)    # (batch_size, time_steps)
        
        # 4. Context vector (weighted sum)
        # alpha unsqueeze: (batch_size, time_steps, 1)
        context = torch.sum(lstm_out * alpha.unsqueeze(-1), dim=1) # (batch_size, hidden_dim)
        return context, alpha

class CBLA_DL_Model(nn.Module):
    def __init__(self, input_nodes=11, time_steps=24):
        super(CBLA_DL_Model, self).__init__()
        
        # 1. 1D-CNN Layer
        # input shape: (batch_size, features=11, time_steps=24)
        # output shape: (batch_size, filters=16, time_steps)
        self.conv1d = nn.Conv1d(in_channels=input_nodes, out_channels=16, kernel_size=1, padding='valid')
        
        # 2. BiLSTM Layer
        # PyTorch LSTM expects input as (batch, seq, feature), so we need to permute before this
        # hidden_units = 8, bidirectional = True -> output feature size = 16
        self.lstm = nn.LSTM(input_size=16, hidden_size=8, batch_first=True, bidirectional=True)
        
        # 3. Attention Layer
        self.attention = AttentionLayer(hidden_dim=16) # 8 * 2 for bidirectional
        
        # 4. Output Layer
        self.fc = nn.Linear(16, 1)

    def forward(self, x):
        # x shape: (batch_size, time_steps=24, features=11)
        
        # Conv1d expects (batch_size, channels, seq_len)
        x = x.permute(0, 2, 1) # (batch_size, 11, 24)
        
        # CNN Feature extraction
        c_out = torch.tanh(self.conv1d(x)) # (batch_size, 16, 24)
        
        # Permute back for LSTM: (batch_size, time_steps=24, features=16)
        c_out = c_out.permute(0, 2, 1)
        
        # BiLSTM
        lstm_out, (h_n, c_n) = self.lstm(c_out) # lstm_out: (batch, 24, 16)
        
        # Attention
        context_vector, attn_weights = self.attention(lstm_out) # (batch, 16)
        
        # Final Output (Preliminary Prediction)
        out = self.fc(context_vector) # (batch, 1)
        
        return out.squeeze(1), attn_weights

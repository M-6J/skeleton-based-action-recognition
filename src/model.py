# src/model.py
import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class PoseGRUClassifier(nn.Module):
    def __init__(self, input_size=51, hidden_size=128, num_classes=5, num_layers=2):
        """
        Args:
            input_size: 17 keypoints × 3 coords = 51
            hidden_size: GRU hidden dimension
            num_classes: 분류할 클래스 개수 (5)
            num_layers: GRU layers (2)
        """
        super(PoseGRUClassifier, self).__init__()
        
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )
        
        # Bidirectional이므로 hidden_size * 2
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, num_keypoints, 3)
               = (batch_size, 30, 17, 3)
        
        Returns:
            logits: (batch_size, num_classes)
        """
        batch_size, seq_len, num_kpts, coords = x.shape
        
        # Reshape: (batch, 30, 17, 3) → (batch, 30, 51)
        x = x.view(batch_size, seq_len, -1)
        
        # GRU: (batch, 30, 51) → (batch, 30, hidden*2)
        gru_out, _ = self.gru(x)
        
        # 마지막 timestep 사용
        last_output = gru_out[:, -1, :]  # (batch, hidden*2)

        # Classification
        logits = self.fc(last_output)  # (batch, num_classes)
        
        return logits


# ==================== Positional Encoding ====================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Positional encoding 계산
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# ==================== Transformer 분류기 ====================
class PoseTransformerClassifier(nn.Module):
    """
    Transformer Encoder for Skeleton-based Action Recognition
    """
    def __init__(self, input_size=51, d_model=128, nhead=4, 
                 num_layers=2, num_classes=10, dropout=0.1):
        """
        Args:
            input_size: 17 keypoints × 3 coords = 51
            d_model: Transformer hidden dimension (nhead로 나누어떨어져야 함)
            nhead: Multi-head attention heads (d_model=128 → nhead=4 추천)
            num_layers: Transformer encoder layers
            num_classes: 분류 클래스 수
            dropout: Dropout rate
        """
        super().__init__()
        
        assert d_model % nhead == 0, f"d_model({d_model}) must be divisible by nhead({nhead})"
        
        # Input projection: (batch, seq_len, 51) → (batch, seq_len, d_model)
        self.embedding = nn.Linear(input_size, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,  # 512
            dropout=dropout,
            activation='relu',
            batch_first=True,
            norm_first=False
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Layer norm
        self.norm = nn.LayerNorm(d_model)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            #nn.Dropout(dropout),
            nn.Linear(d_model, num_classes)
        )
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, num_keypoints, 3)
               = (batch_size, 30, 17, 3)
        Returns:
            logits: (batch_size, num_classes)
        """
        batch_size, seq_len, num_kpts, coords = x.shape
        
        # Flatten keypoints: (batch, 30, 17, 3) → (batch, 30, 51)
        x = x.view(batch_size, seq_len, -1)
        
        # Embedding + Positional encoding
        x = self.embedding(x)  # (batch, 30, d_model)
        x = self.pos_encoder(x)
        
        # Transformer encoding
        x = self.transformer(x)  # (batch, 30, d_model)
        x = self.norm(x)
        
        # Temporal pooling: mean pooling over sequence
        x = x.mean(dim=1)  # (batch, d_model)
        
        # Classification
        logits = self.classifier(x)  # (batch, num_classes)
        
        return logits

# ==================== Temporal Convolutional Network (TCN) ====================
class TemporalBlock(nn.Module):
    """TCN의 기본 블록 (causal convolution + residual)"""
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, dropout=0.1):
        super().__init__()
        
        padding = (kernel_size - 1) * dilation  # Causal padding
        
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, dilation=dilation
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size,
            stride=stride, padding=padding, dilation=dilation
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        
        # Residual connection
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # Causal convolution: trim future timesteps
        out = self.conv1(x)
        out = out[:, :, :-self.conv1.padding[0]]  # Remove future padding
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.dropout1(out)
        
        out = self.conv2(out)
        out = out[:, :, :-self.conv2.padding[0]]
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.dropout2(out)
        
        # Residual
        res = x if self.downsample is None else self.downsample(x)
        
        return self.relu(out + res)


class PoseTCNClassifier(nn.Module):
    """
    Temporal Convolutional Network (TCN)
    """
    def __init__(self, input_size=51, num_channels=[128, 256, 512], 
                 kernel_size=3, num_classes=10, dropout=0.1):
        """
        Args:
            input_size: 17 * 3 = 51
            num_channels: Channel sizes for each TCN layer
            kernel_size: Convolution kernel size
            num_classes: Number of action classes
            dropout: Dropout rate
        """
        super().__init__()
        
        layers = []
        num_levels = len(num_channels)
        
        for i in range(num_levels):
            dilation = 2 ** i  # Exponential dilation: 1, 2, 4, 8...
            in_ch = input_size if i == 0 else num_channels[i-1]
            out_ch = num_channels[i]
            
            layers.append(TemporalBlock(
                in_ch, out_ch, kernel_size, stride=1,
                dilation=dilation, dropout=dropout
            ))
        
        self.network = nn.Sequential(*layers)
        
        # Global pooling + Classification
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(num_channels[-1], 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        # x: (batch, seq_len, num_kpts, coords) = (batch, 30, 17, 3)
        batch_size, seq_len, num_kpts, coords = x.shape
        
        # Flatten: (batch, 30, 51)
        x = x.view(batch_size, seq_len, -1)
        
        # Transpose for Conv1d: (batch, 51, 30)
        x = x.transpose(1, 2)
        
        # TCN layers
        x = self.network(x)  # (batch, channels[-1], seq_len')
        
        # Classification
        x = self.classifier(x)
        
        return x


# ==================== MS-TCN (Multi-Stage TCN) ====================
class PoseMSTCNClassifier(nn.Module):
    """
    Multi-Stage Temporal Convolutional Network
    """
    def __init__(self, input_size=51, num_stages=3, num_layers=2,
                 num_channels=128, num_classes=10, dropout=0.1):
        super().__init__()
        
        self.num_stages = num_stages
        
        # Stage 1: Initial feature extraction
        self.stage1 = nn.Sequential(
            nn.Conv1d(input_size, num_channels, 1),
            nn.ReLU()
        )
        
        # Subsequent stages
        self.stages = nn.ModuleList()
        for s in range(num_stages):
            stage_layers = []
            for i in range(num_layers):
                dilation = 2 ** i
                stage_layers.append(TemporalBlock(
                    num_channels, num_channels, kernel_size=3,
                    stride=1, dilation=dilation, dropout=dropout
                ))
            self.stages.append(nn.Sequential(*stage_layers))
        
        # Classification
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(num_channels, num_classes)
        )
    
    def forward(self, x):
        batch_size, seq_len, num_kpts, coords = x.shape
        
        # Flatten and transpose
        x = x.view(batch_size, seq_len, -1).transpose(1, 2)
        
        # Stage 1
        x = self.stage1(x)
        
        # Multi-stage refinement
        for stage in self.stages:
            x = stage(x)
        
        # Classification
        x = self.classifier(x)
        
        return x
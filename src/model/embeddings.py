# ==============================================================================
# 【嵌入层模块】各类特征嵌入层定义
# ==============================================================================

import torch
import torch.nn as nn


class PitchEmbedding(nn.Module):
    """音高特征嵌入层"""
    def __init__(self, embedding_type, input_dim, hidden_dim):
        super(PitchEmbedding, self).__init__()
        self.embedding_type = embedding_type
        
        if embedding_type == 'cnn':
            self.embedding = nn.Sequential(
                nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv1d(in_channels=32, out_channels=hidden_dim, kernel_size=3, stride=1, padding=1),
                nn.AdaptiveAvgPool1d(1)
            )
        elif embedding_type == 'linear':
            self.embedding = nn.Linear(input_dim, hidden_dim)
    
    def forward(self, pitch_input):
        return self.embedding(pitch_input)


class MFCCEmbedding(nn.Module):
    """MFCC特征嵌入层"""
    def __init__(self, num_mfcc, hidden_dim):
        super(MFCCEmbedding, self).__init__()
        self.embedding = nn.Linear(num_mfcc, hidden_dim)
    
    def forward(self, mfcc_input):
        return self.embedding(mfcc_input)

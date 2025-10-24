# ==============================================================================
# 【注意力融合模块】多种注意力机制和融合策略
# ==============================================================================

import torch
import torch.nn as nn


class FractalAttentionFusion(nn.Module):
    """
    分形自注意力机制融合
    【核心创新】将长序列分割成固定大小的窗口，在每个窗口内独立计算注意力
    """
    def __init__(self, hidden_dim, num_heads, window_size):
        super(FractalAttentionFusion, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.attention_layer = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads)
    
    def forward(self, pitch_embeds, wav2vec_outputs):
        """
        前向传播
        
        Args:
            pitch_embeds: (batch_size, seq_len, hidden_dim)
            wav2vec_outputs: (batch_size, seq_len, hidden_dim)
            
        Returns:
            final_output: (batch_size, hidden_dim)
        """
        batch_size = pitch_embeds.size(0)
        hidden_dim = self.hidden_dim
        
        pitch_seq_len = pitch_embeds.size(1)
        tf_seq_len = wav2vec_outputs.size(1)
        num_windows = min(pitch_seq_len, tf_seq_len) // self.window_size
        
        wav2vec_windows = wav2vec_outputs[:, :num_windows * self.window_size, :].reshape(
            batch_size, num_windows, self.window_size, hidden_dim)
        pitch_windows = pitch_embeds[:, :num_windows * self.window_size, :].reshape(
            batch_size, num_windows, self.window_size, hidden_dim)
        
        attention_outputs = []
        for i in range(num_windows):
            query = pitch_windows[:, i, :, :]
            key = wav2vec_windows[:, i, :, :]
            value = wav2vec_windows[:, i, :, :]
            
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)
            
            attention_output, _ = self.attention_layer(query=query, key=key, value=value)
            attention_outputs.append(attention_output)
        
        attention_outputs = torch.cat(attention_outputs, dim=1)
        
        num_windows = attention_outputs.size(0) // batch_size
        attention_outputs = attention_outputs.view(batch_size, num_windows, -1, hidden_dim)
        
        pooled_output = attention_outputs.mean(dim=2)
        final_output = pooled_output.mean(dim=1)
        
        return final_output


class CrossAttentionFusion(nn.Module):
    """交叉注意力融合机制"""
    def __init__(self, hidden_dim, num_heads):
        super(CrossAttentionFusion, self).__init__()
        self.attention_layer = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads)
    
    def forward(self, query, key, value):
        """
        前向传播
        
        Args:
            query: (seq_len, batch_size, hidden_dim) 或 (batch_size, seq_len, hidden_dim)
            key: 同上
            value: 同上
            
        Returns:
            output: (seq_len, batch_size, hidden_dim) 或 (batch_size, seq_len, hidden_dim)
        """
        attention_output, _ = self.attention_layer(query=query, key=key, value=value)
        return attention_output


class DualDirectionAttentionFusion(nn.Module):
    """双向交叉注意力融合"""
    def __init__(self, hidden_dim, num_heads):
        super(DualDirectionAttentionFusion, self).__init__()
        self.attention_layer = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads)
    
    def forward(self, features1, features2):
        """
        双向注意力融合
        
        Args:
            features1: (batch_size, seq_len, hidden_dim)
            features2: (batch_size, seq_len, hidden_dim)
            
        Returns:
            concatenated_features: (batch_size*2, hidden_dim) 或根据需求调整
        """
        # 第一路：features1 -> features2
        attention_output1, _ = self.attention_layer(query=features1, key=features2, value=features2)
        pooled_output1 = attention_output1.mean(dim=0)
        
        # 第二路：features2 -> features1
        attention_output2, _ = self.attention_layer(query=features2, key=features1, value=features1)
        pooled_output2 = attention_output2.mean(dim=0)
        
        return pooled_output1, pooled_output2

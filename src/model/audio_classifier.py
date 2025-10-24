# ==============================================================================
# 【PCM项目核心模块】IEMOCAP音频模型 - 多模态情感识别模型
# ==============================================================================
# 功能概述：
#   1. 多模态特征融合（音频+文本+音高）
#   2. 10+种任务模式支持（normal/intonation/temporal/ctc等）
#   3. 多种预训练模型集成（Wav2Vec2/RoBERTa/BERT）
#   4. 注意力机制融合（Cross-Attention/Fractal Attention）
#   5. 多任务学习架构（VAD情感维度预测）
# ==============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoders import AudioEncoderFactory, TextEncoderFactory
from .embeddings import PitchEmbedding, MFCCEmbedding
from .attention import FractalAttentionFusion, CrossAttentionFusion, DualDirectionAttentionFusion
from .utils import interpolate_pitch_embeds


class AudioClassifier(torch.nn.Module):
    def __init__(self, args):
        super(AudioClassifier, self).__init__()
        
        self.args = args
        
        # ==================== 文本编码器初始化 ====================
        if self.args.text == "ok":
            text_model, self.t_input_dim = TextEncoderFactory.create_text_encoder(
                self.args.text, self.args.text_model_name
            )
            self.text_model = text_model
        
        # ==================== 音频编码器初始化 ====================
        audio_model, a_hidden_size = AudioEncoderFactory.create_audio_encoder(
            self.args.task, self.args.audio_model_name, 
            getattr(self.args, 'trun_length', None)
        )
        self.audio_model = audio_model
        self.a_hidden_size = a_hidden_size
        
        # ==================== 图像处理层（针对音高轮廓图像） ====================
        if self.args.task in ['contour_image_and_ctc', 'contour_image_vad']:
            self.conv1 = nn.Conv2d(3, 4, kernel_size=3, stride=1, padding=1)
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
            self.conv2 = nn.Conv2d(4, 8, kernel_size=3, stride=1, padding=1)
            self.linear = nn.Linear(4 * 150 * 750, self.a_hidden_size)
        
        # ==================== 基础参数 ====================
        self.num_classes = 3
        self.num_mfcc = 13
        
        # ==================== 任务特定层初始化 ====================
        self._init_task_specific_layers()
    
    def _init_task_specific_layers(self):
        """根据任务类型初始化特定的层"""
        task = self.args.task
        
        if task in ['normal', 'audio_parsing', 'intonation_contour', 'ctc_normal', 
                   'msp_normal', 'temporal_normal']:
            self.rnn = nn.LSTM(
                input_size=1024,
                hidden_size=512,
                num_layers=1,
                batch_first=True,
            )
            self.a_hidden_size = 512
            self.linear = nn.Linear(self.a_hidden_size, self.num_classes)
            
        elif task in ['temporal_intonation_and_wav2vec2']:
            self.window_size = 32
            self.attention_layer = nn.MultiheadAttention(
                embed_dim=self.a_hidden_size, num_heads=8
            )
            pitch_dim = 1
            self.pitch_embedding = nn.Linear(pitch_dim, self.a_hidden_size)
            self.linear = nn.Linear(self.a_hidden_size, self.num_classes)
            
        elif task in ['audio_parsing_probing']:
            self.linear = nn.Linear(self.t_input_dim, self.num_classes)
            
        elif task in ['normal_with_text', 'audio_parsing_with_text_concat']:
            self.linear = nn.Linear(self.a_hidden_size + self.t_input_dim, self.num_classes)
            self.attention_layer = nn.MultiheadAttention(
                embed_dim=self.a_hidden_size, num_heads=8
            )
            self.fc = nn.Linear(self.a_hidden_size, self.num_classes)
            
        elif task in ['audio_parsing_with_text']:
            sentence_len = 10
            self.max_len = (self.a_hidden_size + self.t_input_dim) * sentence_len
            self.linear = nn.Linear(self.max_len, self.num_classes).cuda()
            
        elif task in ['intonation_and_wav2vec2', 'interpolated_intonation_and_wav2vec2',
                     'intonation_contour_with_text', 'msp_intonation_and_wav2vec2']:
            self._init_pitch_fusion_layers()
            
        elif task in ['mfcc_wav2vec2']:
            self.attention_layer = nn.MultiheadAttention(
                embed_dim=self.a_hidden_size, num_heads=8
            )
            self.mfcc_embedding = MFCCEmbedding(self.num_mfcc, self.a_hidden_size)
            self.fc = nn.Linear(self.a_hidden_size, self.num_classes)
            
        elif task in ['double_wav2vec2']:
            self.attention_layer = nn.MultiheadAttention(
                embed_dim=self.a_hidden_size, num_heads=8
            )
            self.fc = nn.Linear(self.a_hidden_size, self.num_classes)
            
        elif task in ['contour_image_and_ctc', 'contour_image_vad']:
            self.attention_layer = nn.MultiheadAttention(
                embed_dim=self.a_hidden_size, num_heads=4
            )
            self.fc = nn.Linear(self.a_hidden_size * 2, self.num_classes)
    
    def _init_pitch_fusion_layers(self):
        """初始化音高融合相关层"""
        if self.args.embedding == 'cnn':
            self.pitch_embedding = PitchEmbedding('cnn', 1, self.a_hidden_size)
        elif self.args.embedding == 'linear':
            self.pitch_embedding = PitchEmbedding('linear', self.a_hidden_size, self.a_hidden_size)
        
        self.attention_layer = nn.MultiheadAttention(
            embed_dim=self.a_hidden_size, num_heads=8
        )
        self.fc = nn.Linear(self.a_hidden_size, self.num_classes)

    def freeze_prameters(self):
        self.audio_model.feature_extractor._freeze_parameters()
    
    def get_hidden_states(self):
        return self.hidden_states
    
    def forward(self, pitch_audio, audio_inputs, text_input_tokens, text_attention_mask):
        """多模态前向传播主函数"""
        
        if self.args.task in ['normal', 'audio_parsing', 'msp_normal']:
            return self._forward_normal(audio_inputs)
        
        elif self.args.task in ['temporal_normal']:
            return self._forward_temporal_normal(audio_inputs, text_attention_mask)
        
        elif self.args.task in ['temporal_intonation_and_wav2vec2']:
            return self._forward_temporal_pitch_fusion(pitch_audio, audio_inputs)
        
        elif self.args.task in ['ctc_normal']:
            return self._forward_ctc(audio_inputs)
        
        elif self.args.task in ['contour_image_and_ctc']:
            return self._forward_contour_image_ctc(audio_inputs, text_attention_mask)
        
        elif self.args.task in ['contour_image_vad']:
            return self._forward_contour_image_vad(audio_inputs, text_attention_mask)
        
        elif self.args.task in ['intonation_contour']:
            return self._forward_intonation_contour(audio_inputs)
        
        elif self.args.task in ['normal_with_text']:
            return self._forward_normal_with_text(audio_inputs, text_input_tokens, text_attention_mask)
        
        elif self.args.task in ['audio_parsing_probing']:
            return self._forward_audio_parsing_probing(text_input_tokens, text_attention_mask)
        
        elif self.args.task in ['audio_parsing_with_text']:
            return self._forward_audio_parsing_with_text(audio_inputs, text_input_tokens, text_attention_mask)
        
        elif self.args.task in ['audio_parsing_with_text_concat']:
            return self._forward_audio_parsing_with_text_concat(audio_inputs, text_input_tokens, text_attention_mask)
        
        elif self.args.task in ['intonation_contour_with_text']:
            return self._forward_intonation_contour_with_text(pitch_audio, audio_inputs, text_input_tokens, text_attention_mask)
        
        elif self.args.task in ['intonation_and_wav2vec2', 'msp_intonation_and_wav2vec2', 
                               'interpolated_intonation_and_wav2vec2']:
            return self._forward_pitch_fusion(pitch_audio, audio_inputs)
        
        elif self.args.task in ['mfcc_wav2vec2']:
            return self._forward_mfcc_fusion(pitch_audio, audio_inputs)
        
        elif self.args.task in ['double_wav2vec2']:
            return self._forward_double_wav2vec2(audio_inputs)
        
        else:
            raise ValueError(f"Unknown task: {self.args.task}")
    
    def _forward_normal(self, audio_inputs):
        """【标准模式】基础音频特征提取"""
        if self.args.layer == "last":
            hidden_states = self.audio_model(audio_inputs, output_hidden_states=True).last_hidden_state
        else:
            layer_num = self.args.layer
            hidden_states = self.audio_model(audio_inputs, output_hidden_states=True).hidden_states[layer_num]
        
        audio_output = hidden_states.mean(dim=1)
        output = self.linear(audio_output)
        return output
    
    def _forward_temporal_normal(self, audio_inputs, text_attention_mask):
        """【时序模式】使用LSTM处理时序特征"""
        hidden_states = self.audio_model(audio_inputs, attention_mask=text_attention_mask, 
                                        output_hidden_states=True).last_hidden_state
        rnn_output, _ = self.rnn(hidden_states)
        final = rnn_output[:, -1, :]
        output = self.linear(final)
        return output
    
    def _forward_temporal_pitch_fusion(self, pitch_audio, audio_inputs):
        """【时序音高融合模式】分形自注意力机制"""
        batch_size = 1
        hidden_dim = self.a_hidden_size
        
        pitches = pitch_audio
        tf_audio = audio_inputs
        wav2vec_outputs = self.audio_model(tf_audio, output_hidden_states=True)
        self.hidden_states = wav2vec_outputs.hidden_states
        
        wav2vec_outputs = self.hidden_states[-1]
        pitches = pitches.to(dtype=torch.float32)
        pitch_embeds = self.pitch_embedding(pitches)
        
        target_seq_len = 299
        pitch_embeds = interpolate_pitch_embeds(pitch_embeds, target_seq_len)
        
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
        
        output = self.linear(final_output)
        return output
    
    def _forward_ctc(self, audio_inputs):
        """【CTC模式】基于连接时序分类"""
        hidden_states = self.audio_model(audio_inputs, output_hidden_states=True).hidden_states[12]
        audio_output = hidden_states.mean(dim=1)
        vad_output = self.linear(audio_output)
        output = (hidden_states, vad_output)
        return output
    
    def _forward_contour_image_ctc(self, audio_inputs, text_attention_mask):
        """【音高轮廓图像+CTC模式】"""
        audio_image = self.pool(F.relu(self.conv1(text_attention_mask)))
        audio_image = self.pool(F.relu(self.conv2(audio_image)))
        audio_image = audio_image.view(audio_image.size(0), -1)
        processed_audio_image = self.linear(audio_image)
        
        hidden_states = self.audio_model(audio_inputs, output_hidden_states=True).hidden_states[12]
        w2v2_features = hidden_states.mean(dim=1)
        
        attention_output, _ = self.attention_layer(query=processed_audio_image, 
                                                   key=w2v2_features, value=w2v2_features)
        pooled_output = attention_output.mean(dim=0)
        vad_output = self.fc(pooled_output)
        
        output = (hidden_states, vad_output)
        return output
    
    def _forward_contour_image_vad(self, audio_inputs, text_attention_mask):
        """【音高轮廓图像+VAD模式】双向注意力融合"""
        audio_image = self.pool(F.relu(self.conv1(text_attention_mask)))
        audio_image = self.pool(F.relu(self.conv2(audio_image)))
        audio_image = audio_image.view(audio_image.size(0), -1)
        processed_audio_image = self.linear(audio_image)
        
        hidden_states = self.audio_model(audio_inputs, output_hidden_states=True).hidden_states[12]
        w2v2_features = hidden_states.mean(dim=1)
        
        attention_output, _ = self.attention_layer(query=processed_audio_image, 
                                                   key=w2v2_features, value=w2v2_features)
        attention_output2, _2 = self.attention_layer(query=w2v2_features, 
                                                     key=processed_audio_image, 
                                                     value=processed_audio_image)
        
        pooled_output1 = attention_output.mean(dim=0)
        pooled_output2 = attention_output2.mean(dim=0)
        
        concatenated_features = torch.cat((pooled_output1, pooled_output2), dim=0)
        output = self.fc(concatenated_features)
        return output
    
    def _forward_intonation_contour(self, audio_inputs):
        """【音高轮廓模式】直接使用音高统计特征"""
        audio_output = audio_inputs
        output = self.linear(audio_output)
        return output
    
    def _forward_normal_with_text(self, audio_inputs, text_input_tokens, text_attention_mask):
        """【文本+音频融合模式】RoBERTa + Wav2Vec2"""
        last_token = self.text_model(input_ids=text_input_tokens, 
                                    attention_mask=text_attention_mask)['last_hidden_state']
        
        if self.args.layer == "last":
            hidden_states = self.audio_model(audio_inputs, output_hidden_states=True).last_hidden_state
        elif self.args.layer == "layer7":
            hidden_states = self.audio_model(audio_inputs, output_hidden_states=True).hidden_states[7]
        
        last_token = last_token.to(torch.float32)
        hidden_states = hidden_states.to(torch.float32)
        
        last_token = last_token.permute(1, 0, 2)
        hidden_states = hidden_states.permute(1, 0, 2)
        
        attention_output, _ = self.attention_layer(query=hidden_states, 
                                                   key=last_token, value=last_token)
        
        pooled_output = attention_output.mean(dim=0)
        output = self.fc(pooled_output)
        return output
    
    def _forward_audio_parsing_probing(self, text_input_tokens, text_attention_mask):
        """【音频解析探针模式】"""
        logits = self.text_model(input_ids=text_input_tokens, 
                                attention_mask=text_attention_mask)['last_hidden_state']
        last_token = logits.mean(dim=1)
        output = self.linear(last_token)
        return output
    
    def _forward_audio_parsing_with_text(self, audio_inputs, text_input_tokens, text_attention_mask):
        """【音频解析+文本模式】"""
        first_outs = self.text_model(input_ids=text_input_tokens, 
                                     attention_mask=text_attention_mask)['last_hidden_state']
        last_token = first_outs.mean(dim=1)
        
        if self.args.layer == "last":
            hidden_states = self.audio_model(audio_inputs).last_hidden_state
        elif self.args.layer == "layer7":
            hidden_states = self.audio_model(audio_inputs, 
                                           output_hidden_states=True).hidden_states[7]
        audio_output = hidden_states.mean(dim=1)
        
        last_token = last_token.to(torch.float32)
        audio_output = audio_output.to(torch.float32)
        
        audio_output = torch.cat((audio_output, last_token), dim=1)
        audio_output = audio_output.view(-1)
        
        if audio_output.shape[0] > self.max_len:
            padded_tensor = audio_output[:self.max_len]
        elif audio_output.shape[0] < self.max_len:
            padding_size = self.max_len - audio_output.shape[0]
            padded_tensor = F.pad(audio_output, (0, padding_size), 'constant', 0)
        else:
            padded_tensor = audio_output
        
        output = self.linear(padded_tensor)
        return output
    
    def _forward_audio_parsing_with_text_concat(self, audio_inputs, text_input_tokens, text_attention_mask):
        """【音频解析+文本拼接模式】"""
        first_outs = self.text_model(input_ids=text_input_tokens, 
                                     attention_mask=text_attention_mask)['last_hidden_state']
        last_token = first_outs.mean(dim=1)
        
        if self.args.layer == "last":
            hidden_states = self.audio_model(audio_inputs).last_hidden_state
        elif self.args.layer == "layer7":
            hidden_states = self.audio_model(audio_inputs, 
                                           output_hidden_states=True).hidden_states[6]
        audio_output = hidden_states.mean(dim=1)
        
        last_token = last_token.to(torch.float32)
        audio_output = audio_output.to(torch.float32)
        
        concatenated_features = torch.cat((audio_output, last_token), dim=1)
        output = self.linear(concatenated_features)
        return output
    
    def _forward_intonation_contour_with_text(self, pitch_audio, audio_inputs, text_input_tokens, text_attention_mask):
        """【音高轮廓+文本模式】"""
        last_token = self.text_model(input_ids=text_input_tokens, 
                                    attention_mask=text_attention_mask)['last_hidden_state']
        
        last_token = last_token.to(torch.float32)
        audio_inputs = audio_inputs.to(torch.float32)
        
        pitch_embeds = self.pitch_embedding(audio_inputs)
        
        last_token = last_token.permute(1, 0, 2)
        attention_output, _ = self.attention_layer(query=pitch_embeds, 
                                                   key=last_token, value=last_token)
        
        pooled_output = attention_output.mean(dim=0)
        output = self.fc(pooled_output)
        return output
    
    def _forward_pitch_fusion(self, pitch_audio, audio_inputs):
        """【PCM核心模式】音高+Wav2Vec2双模态融合"""
        pitches = pitch_audio
        tf_audio = audio_inputs
        
        wav2vec_outputs = self.audio_model(tf_audio, output_hidden_states=True)
        self.hidden_states = wav2vec_outputs.hidden_states
        wav2vec_outputs = self.hidden_states[-1]
        
        if self.args.embedding == 'linear':
            pitch_embeds = self.pitch_embedding(pitches)
        elif self.args.embedding == 'cnn':
            if self.args.dataset == 'iemocap':
                pitches = pitch_audio.unsqueeze(1)
            elif self.args.dataset == 'msp_podcast':
                pitches = pitches.squeeze(-1).unsqueeze(1)
            pitches = pitches.to(dtype=torch.float32)
            pitch_embeds = self.pitch_embedding(pitches)
        
        if self.args.dataset == 'iemocap':
            pitch_embeds = pitch_embeds.squeeze(-1)
            pitch_embeds = pitch_embeds.unsqueeze(1).expand(-1, wav2vec_outputs.size(1), -1)
        elif self.args.dataset == 'msp_podcast':
            pitch_embeds = pitch_embeds.transpose(1, 2)
            pitch_embeds = pitch_embeds.expand(-1, wav2vec_outputs.size(1), -1)
        
        attention_output, _ = self.attention_layer(query=pitch_embeds, 
                                                   key=self.hidden_states[-1], 
                                                   value=self.hidden_states[-1])
        
        pooled_output = attention_output.mean(dim=1)
        output = self.fc(pooled_output)
        return output
    
    def _forward_mfcc_fusion(self, pitch_audio, audio_inputs):
        """【MFCC+Wav2Vec2融合模式】"""
        mfcc = pitch_audio
        tf_audio = audio_inputs
        
        wav2vec_outputs = self.audio_model(tf_audio).last_hidden_state
        mfcc_embeds = self.mfcc_embedding(mfcc)
        
        if mfcc_embeds.size(1) != wav2vec_outputs.size(1):
            target_length = wav2vec_outputs.size(1)
            mfcc_embeds = torch.nn.functional.interpolate(
                mfcc_embeds.permute(0, 2, 1), size=target_length
            ).permute(0, 2, 1)
        
        attention_output, _ = self.attention_layer(query=wav2vec_outputs, 
                                                   key=mfcc_embeds, 
                                                   value=wav2vec_outputs)
        
        pooled_output = attention_output.mean(dim=1)
        output = self.fc(pooled_output)
        return output
    
    def _forward_double_wav2vec2(self, audio_inputs):
        """【双Wav2Vec2层融合模式】"""
        wav2vec_outputs = self.audio_model(audio_inputs, output_hidden_states=True)
        
        layer12 = wav2vec_outputs.last_hidden_state
        layer7 = wav2vec_outputs.hidden_states[7]
        
        attention_output, _ = self.attention_layer(query=layer7, key=layer12, value=layer12)
        
        pooled_output = attention_output.mean(dim=1)
        output = self.fc(pooled_output)
        return output

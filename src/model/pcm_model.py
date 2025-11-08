import torch
from transformers import Wav2Vec2Model
import torch.nn as nn

class AudioClassifier(nn.Module):

    def __init__(self, args):
        super(AudioClassifier, self).__init__()
        self.args = args
        self.audio_model = Wav2Vec2Model.from_pretrained(self.args.audio_model_name, mask_feature_length=10).cuda()
        self.a_hidden_size = self.audio_model.config.hidden_size
        self.num_classes = 3  

        # 与配置/数据集保持一致：使用 basemodel 作为纯 Wav2Vec2 线性头分支
        if self.args.exp_name == "basemodel":
            self.linear = torch.nn.Linear(self.a_hidden_size, self.num_classes)
        elif self.args.exp_name =='cnn':
                self.pitch_embedding = nn.Sequential(
                    nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(),
                    nn.Conv1d(in_channels=32, out_channels=self.a_hidden_size, kernel_size=3, stride=1, padding=1),
                    # 自适应平均池化沿时序取全局均值，输出 shape 变为 (batch, hidden_size, 1)
                    nn.AdaptiveAvgPool1d(1)
                )

        elif self.args.exp_name.startswith('linear'):
            self.pitch_embedding = torch.nn.Linear(self.a_hidden_size, self.a_hidden_size)

        # 注意：使用 batch_first=True，后续传入张量统一为 (B, S, E)
        self.attention_layer = torch.nn.MultiheadAttention(
            embed_dim=self.a_hidden_size, num_heads=8, batch_first=True
        )

        self.fc = torch.nn.Linear(self.a_hidden_size, self.num_classes)

    def freeze_prameters(self,):
        self.audio_model.feature_extractor._freeze_parameters()

    def forward(self, pitch_features, tf_audio):

        wav2vec_outputs = self.audio_model(tf_audio, output_hidden_states=True).last_hidden_state

        if self.args.exp_name == "basemodel":
            audio_output = wav2vec_outputs.mean(dim=1)
            output = self.linear(audio_output)
        elif self.args.exp_name == 'cnn':
            # 期望 pitch_features 形状为 (B, L) 或 (B, L, 1)，据此构造成 Conv1d 输入 (B, 1, L)
            if pitch_features.dim() == 3 and pitch_features.size(-1) == 1:
                # (B, L, 1) -> (B, 1, L)
                pitches = pitch_features.squeeze(-1).unsqueeze(1)
            elif pitch_features.dim() == 2:
                # (B, L) -> (B, 1, L)
                pitches = pitch_features.unsqueeze(1)
            else:
                # 兜底：尽量挤压到 (B, 1, L)
                pitches = pitch_features
                pitches = pitches.view(pitches.size(0), 1, -1)

            pitches = pitches.to(dtype=torch.float32)
            # Conv1d 输出 (B, hidden, L'); 自适应池化到 L'=1 -> (B, hidden, 1)
            pitch_embeds = self.pitch_embedding(pitches)
            # 去掉长度维，得到 (B, hidden)
            pitch_embeds = pitch_embeds.squeeze(-1)
            # 扩展到与 wav2vec2 序列长度一致，得到 (B, S, E)
            pitch_embeds = pitch_embeds.unsqueeze(1).expand(-1, wav2vec_outputs.size(1), -1)

            # 注意：MultiheadAttention 已设置 batch_first=True，输入 (B, S, E)
            attention_output, _ = self.attention_layer(
                query=pitch_embeds, key=wav2vec_outputs, value=wav2vec_outputs
            )
            pooled_output = attention_output.mean(dim=1)
            output = self.fc(pooled_output)
        elif self.args.exp_name.startswith('linear'):
            pitch_embeds = self.pitch_embedding(pitch_features)
            pitch_embeds = pitch_embeds.unsqueeze(1).expand(-1, wav2vec_outputs.size(1), -1)

            attention_output, _ = self.attention_layer(query=pitch_embeds, key=wav2vec_outputs, value=wav2vec_outputs)
            pooled_output = attention_output.mean(dim=1)
            output = self.fc(pooled_output)

        return output

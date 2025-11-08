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

        if self.args.exp_name == "basename":
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

        self.attention_layer = torch.nn.MultiheadAttention(embed_dim=self.a_hidden_size, num_heads=8)

        self.fc = torch.nn.Linear(self.a_hidden_size, self.num_classes)

    def freeze_prameters(self,):
        self.audio_model.feature_extractor._freeze_parameters()

    def forward(self, pitch_features, tf_audio):

        wav2vec_outputs = self.audio_model(tf_audio, output_hidden_states=True).last_hidden_state

        if self.args.exp_name == "basename":
            audio_output = wav2vec_outputs.mean(dim=1)
            output = self.linear(audio_output)
        elif self.args.exp_name == 'cnn':
            tf_audio = tf_audio
            pitch_features = pitch_features
            pitches = pitch_features.unsqueeze(1)
            pitches = pitches.to(dtype=torch.float32)
            pitch_embeds = self.pitch_embedding(pitches)
            pitch_embeds = pitch_embeds.unsqueeze(1).expand(-1, wav2vec_outputs.size(1), -1)
            attention_output, _ = self.attention_layer(query=pitch_embeds, key=wav2vec_outputs, value=wav2vec_outputs)
            pooled_output = attention_output.mean(dim=1)
            output = self.fc(pooled_output)
        elif self.args.exp_name.startswith('linear'):
            pitch_embeds = self.pitch_embedding(pitch_features)
            pitch_embeds = pitch_embeds.unsqueeze(1).expand(-1, wav2vec_outputs.size(1), -1)

            attention_output, _ = self.attention_layer(query=pitch_embeds, key=wav2vec_outputs, value=wav2vec_outputs)
            pooled_output = attention_output.mean(dim=1)
            output = self.fc(pooled_output)

        return output
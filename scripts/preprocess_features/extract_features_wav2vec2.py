import argparse
import os
from tqdm import tqdm
import torch
import torchaudio
import numpy as np
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

def extract_features(audio_file, feature_extractor, model, layer_idx, device):
    
    waveform, sample_rate = torchaudio.load(audio_file)

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)

    # 归一化
    input_values = feature_extractor(waveform.squeeze().numpy(), sampling_rate=16000, return_tensors="pt").input_values
    input_values = input_values.to(device)
    with torch.no_grad():
        outputs = model(input_values, output_hidden_states=True)
        features = outputs.hidden_states[layer_idx]

    return features

def save_features(features, save_path):
    np.save(save_path, features.cpu().numpy())

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        default="data/raw/IEMOCAP_full_release",
        help="输入文件夹的路径"
                        )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/features/iemocap",
        help="输出文件夹的路径"
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=12,
        help="要提取的特征层数"
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="pretrained_model/wav2vec2-large-robust",
        help="预训练模型的路径"
    )
    parser.add_argument(
        "--gpu_id",
        type=int,
        default=0,
        help="指定使用哪一块GPU,默认0"
    )

    args = parser.parse_args()

    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(args.model_dir)
    model = Wav2Vec2Model.from_pretrained(args.model_dir)
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    audio_files = []
    for root, dirs, files in os.walk(args.input):
        for file in files:
            if file.lower().endswith(('.wav', '.mp3')):
                audio_path = os.path.join(root, file)
                audio_files.append(audio_path)

    with tqdm(audio_files, desc="提取音频特征") as pbar:
        for audio_path in pbar:
            rel_path = os.path.relpath(audio_path, args.input)
            model_name = os.path.basename(args.model_dir)
            save_dir = os.path.join(args.output, f"{model_name}_l{args.layer}", os.path.dirname(rel_path))
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, os.path.splitext(os.path.basename(audio_path))[0] + '.npy')
            features = extract_features(audio_path, feature_extractor, model, args.layer, device)
            save_features(features, save_path)
            pbar.set_postfix({"shape": tuple(features.shape)})
    
    print(f"完成特征提取, 结果保存在 {args.output}")



if __name__ == "__main__":
    main()

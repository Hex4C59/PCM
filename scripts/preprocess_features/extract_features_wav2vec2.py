import argparse
import os
from tqdm import tqdm
import torch
import torchaudio
import soundfile as sf
import numpy as np
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
from pathlib import Path

TARGET_SAMPLE_RATE = 16000


def _load_waveform(audio_file):
    waveform, sample_rate = sf.read(audio_file, dtype="float32")
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)
    waveform = torch.from_numpy(waveform).unsqueeze(0)
    if sample_rate != TARGET_SAMPLE_RATE:
        waveform = torchaudio.functional.resample(
            waveform, sample_rate, TARGET_SAMPLE_RATE
        )
        sample_rate = TARGET_SAMPLE_RATE
    return waveform, sample_rate


def extract_features(audio_file, feature_extractor, model, layer_idx, device):
    waveform, sample_rate = _load_waveform(audio_file)

    input_values = feature_extractor(
        waveform.squeeze().numpy(), sampling_rate=sample_rate, return_tensors="pt"
    ).input_values
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
        default="data/raw/iemocap_audio",
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

    model_name = os.path.basename(args.model_dir)
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(args.model_dir)
    model = Wav2Vec2Model.from_pretrained(args.model_dir)
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    audio_files = []
    for root, dirs, files in os.walk(args.input):
        # # 路径字符串中同时包含"sentences"和"wav"这两个子字符串
        # # /data/sentences/wav/audio.wav - 通过（包含sentences和wav）
        # if "sentences" in root and "wav" in root:
        for file in files:
            if file.lower().endswith(('.wav', '.mp3')):
                audio_path = os.path.join(root, file)
                audio_files.append(audio_path)

    with tqdm(audio_files, desc="提取音频特征") as pbar:
        for audio_path in pbar:
            # 获取Session名
            parts = Path(audio_path).parts
            session_name = None
            for part in parts:
                if part.startswith("Session"):
                    session_name = part
                    break
            if session_name is None:
                continue

            # 只保留文件名
            file_name = os.path.splitext(os.path.basename(audio_path))[0] + '.npy'
            save_dir = os.path.join(args.output, model_name, session_name)
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, file_name)

            features = extract_features(audio_path, feature_extractor, model, args.layer, device)
            save_features(features, save_path)
            pbar.set_postfix({"shape": tuple(features.shape)})
    
    print(f"完成特征提取, 结果保存在 {args.output}")



if __name__ == "__main__":
    main()

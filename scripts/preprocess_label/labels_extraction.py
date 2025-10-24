import re
import os
import pandas as pd

useful_regex = re.compile(r'\[.+\]\n', re.IGNORECASE)

info_line = re.compile(r'\[.+\]\n', re.IGNORECASE)

def main():
    start_times, end_times, wav_file_names, emotions, vals, acts, doms = [], [], [], [], [], [], []
    df_iemocap = pd.DataFrame(columns=['start_time', 'end_time', 'wav_file', 'emotion', 'val', 'act', 'dom'])
    
    for sess in [5]:
        emo_evaluation_dir = '/mnt/shareEEx/liuyang/code/PCM/data/IEMOCAP/Session{}/dialog/EmoEvaluation/'.format(sess)
        evaluation_files = [l for l in os.listdir(emo_evaluation_dir) if 'Ses' in l]
        for file in evaluation_files:
            with open(emo_evaluation_dir + file) as f:
                content = f.read()
            info_lines = re.findall(info_line, content)
            for line in info_lines[1:]:  # the first line is a header
                start_end_time, wav_file_name, emotion, val_act_dom = line.strip().split('\t')
                start_time, end_time = start_end_time[1:-1].split('-')
                val, act, dom = val_act_dom[1:-1].split(',')
                val, act, dom = float(val), float(act), float(dom)
                start_time, end_time = float(start_time), float(end_time)
                start_times.append(start_time)
                end_times.append(end_time)
                wav_file_names.append(wav_file_name)
                emotions.append(emotion)
                vals.append(val)
                acts.append(act)
                doms.append(dom)
                
    df_iemocap['start_time'] = start_times
    df_iemocap['end_time'] = end_times
    df_iemocap['wav_file'] = wav_file_names
    df_iemocap['emotion'] = emotions
    df_iemocap['val'] = vals
    df_iemocap['act'] = acts
    df_iemocap['dom'] = doms

    df_iemocap.tail()

    df_iemocap.to_csv('/mnt/shareEEx/liuyang/code/PCM/data/labels/iemocap.csv', index=False)

if __name__ == '__main__':
    main()
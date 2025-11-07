import numpy as np
import re
from pathlib import Path
from parselmouth.praat import call


class PitchFeatures:
    def __init__(self, sound, temp_dir = "./tmp_pitch"):
        self.sound = sound
        self.pitch_tier = None
        self.total_duration = None
        self.pitch_point = []
        self.time_point = []

        self.pd = [] # pd（pitch distance）：相邻点的欧氏距离，描述全局变化幅度。
        self.pt = [] # pt（time proportion）：相邻时间点间隔占整段音频总时长的比例，用于把时间差规范到 0–1 范围，便于不同长度样本可比。
        self.ps = [] # ps（slope）：相邻点的斜率，等于频率变化量除以时间差，描述音高上升/下降的速度。
        self.pr = [] # pr（range ratio）：相邻点的音高变化量，相对于整段音高范围（max-min）的比例，反映局部波动在整体音域里所占的相对大小。

        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.temp_file = self.temp_dir / "PitchTier"
    
    def get_pitch_tiers(self,):
        # 调用 Praat 的 "To Manipulation" 方法，
        # 将音频对象（self.sound）转换为 Manipulation 对象，用于后续的语音分析（如音高、时长等操作）。
        manipulation = call(self.sound, "To Manipulation", 0.01, 75, 600)
        self.pitch_tier = call(manipulation, "Extract pitch tier")
        return self.pitch_tier
        
    def stylize_pitch(self,):
        if self.pitch_tier is not None:
            call(self.pitch_tier, "Stylize...",2.0,"semitones")
            tmp_pitch_point = self.pitch_point
            tmp_time_point = self.time_point
            self.set_time_and_pitch_point()
            if len(self.pitch_point) == 0:
                self.pitch_point = tmp_pitch_point
                self.time_point = tmp_time_point 
        else:
            print("pitch_tier is None")
            return 
        
    def set_total_duration(self,):

        total_duration_match = re.search(r'Total duration: (\d+(\.\d+)?) seconds', str(self.pitch_tier))
        if total_duration_match:
            self.total_duration = float(total_duration_match.group(1))
        else:

            print("Total duration not found.")
            
    def set_time_and_pitch_point(self,):
        self.pitch_tier.save(self.temp_file)
        r_file = open(self.temp_file, 'r')
        
        self.pitch_point = []
        self.time_point = []
        while True:
            line = r_file.readline()
            if not line:
                break

            if 'number' in line:
                value = re.sub(r'[^0-9^.]', '', line)

                if value.count('.') > 1:
                    parts = value.split('.')
                    value = parts[0] + ''.join(parts[1:])
                if value != '':
                    self.time_point.append(round(float(value), 4))
            elif 'value' in line:
                value = re.sub(r'[^0-9^.]', '', line)

                if value.count('.') > 1:
                    parts = value.split('.')
                    value = parts[0] + ''.join(parts[1:])
                if value != '':
                    self.pitch_point.append(round(float(value), 4))
                
        if len(self.pitch_point)==0:
            while True:
                line = r_file.readline()
                if not line:
                    break
        r_file.close()
    
    def extract_pd(self,):
        for i in range(len(self.pitch_point) - 1):
            point1 = (self.time_point[i], self.pitch_point[i])
            point2 = (self.time_point[i+1], self.pitch_point[i+1])
            distance = np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
            self.pd.append(round(distance,4))

    def extract_pt(self,):
        for i in range(len(self.time_point)-1):
            gap = self.time_point[i+1]-self.time_point[i]
            percent = round((gap/self.total_duration),4)
            self.pt.append(percent)

    def extract_pr(self,):
        if len(self.pitch_point) == 0:
            return
        total = max(self.pitch_point) - min(self.pitch_point)
        for i in range(1, len(self.pitch_point)):
            delta_y = abs(self.pitch_point[i] - self.pitch_point[i-1])
            
            r = round(delta_y / total,4)
            self.pr.append(r) 
    
    def extract_ps(self,):
        for i in range(1, len(self.time_point)):
            delta_x = self.time_point[i] - self.time_point[i-1]
            delta_y = self.pitch_point[i] - self.pitch_point[i-1]
            
            if delta_x == 0:
                self.ps.append(None)
            else:
                slope = delta_y / delta_x
                self.ps.append(round(slope,4))
                
    def get_features(self,):
        
        self.pitch_tiers = self.get_pitch_tiers()
        self.set_time_and_pitch_point() 
        self.stylize_pitch()

        self.set_total_duration()
        feature = []

        self.extract_pd()
        self.extract_pt()
        self.extract_ps()
        self.extract_pr()
        
        for t,d,s,r in zip(self.pt, self.pd, self.ps,self.pr):
            td, rs =  t * d , r * s
            if s < 0:
                rs = np.sqrt(abs(rs))
                rs = round(-1 * rs, 2)
            td = np.sqrt(td)
            feature.append(td)
            feature.append(rs)
        return feature
    
    def get_pitchs(self,):
        self.pitch_tiers = self.get_pitch_tiers()
        self.set_time_and_pitch_point() 
        return self.pitch_point, self.time_point

    def get_intensity(self,):
        intensity = call(self.sound, "To Intensity...", 100.0, 0.0)
        return intensity.values[0]


    
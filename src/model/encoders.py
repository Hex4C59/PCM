# ==============================================================================
# 【编码器模块】多模态编码器初始化和管理
# ==============================================================================

from transformers import (
    Wav2Vec2Processor, Wav2Vec2Model, Wav2Vec2ForCTC,
    HubertModel, WhisperModel, RobertaModel, BertModel
)


class AudioEncoderFactory:
    """音频编码器工厂类"""
    
    @staticmethod
    def create_audio_encoder(task, audio_model_name, trun_length=None):
        """
        创建音频编码器
        
        Args:
            task: 任务类型
            audio_model_name: 预训练模型名称
            trun_length: 时序模式下的截断长度
            
        Returns:
            (audio_model, hidden_size) 元组
        """
        if task in ['normal', 'msp_normal', 'msp_intonation_and_wav2vec2', 'audio_parsing', 
                   'double_wav2vec2', 'normal_with_text', 'audio_parsing_with_text',
                   'audio_parsing_with_text_concat', 'intonation_and_wav2vec2',
                   'interpolated_intonation_and_wav2vec2', 'mfcc_wav2vec2']:
            model = Wav2Vec2Model.from_pretrained(audio_model_name, mask_feature_length=10).cuda()
            hidden_size = model.config.hidden_size
            
        elif task in ['temporal_normal']:
            model = Wav2Vec2Model.from_pretrained(audio_model_name, mask_feature_length=10)
            hidden_size = trun_length
            
        elif task in ['temporal_intonation_and_wav2vec2']:
            model = Wav2Vec2Model.from_pretrained(audio_model_name, mask_feature_length=10)
            hidden_size = trun_length
            
        elif task in ['ctc_normal']:
            model = Wav2Vec2ForCTC.from_pretrained(audio_model_name).cuda()
            hidden_size = model.config.hidden_size
            
        elif task in ['contour_image_and_ctc', 'contour_image_vad']:
            model = Wav2Vec2ForCTC.from_pretrained(audio_model_name).cuda()
            hidden_size = model.config.hidden_size
            
        elif task in ['intonation_contour']:
            model = None
            hidden_size = 50
            
        elif task in ['intonation_contour_with_text']:
            model = None
            hidden_size = 768  # text_model hidden_size
            
        else:
            raise ValueError(f"Unknown task: {task}")
            
        return model, hidden_size


class TextEncoderFactory:
    """文本编码器工厂类"""
    
    @staticmethod
    def create_text_encoder(use_text, text_model_name):
        """
        创建文本编码器
        
        Args:
            use_text: 是否使用文本模型
            text_model_name: 预训练模型名称
            
        Returns:
            (text_model, input_dim) 元组
        """
        if use_text == "ok":
            text_model = RobertaModel.from_pretrained(text_model_name).cuda()
            input_dim = text_model.config.hidden_size
            return text_model, input_dim
        else:
            return None, 0

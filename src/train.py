import torch
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import yaml
from model.pcm_model import AudioClassifier
from data.dataset import IEMOCAP_Dataset
from torch.utils.data import DataLoader
import pandas as pd
import random
import os 
import torch.nn as nn
import numpy as np
import logging
import argparse
import gc
from src.metrics.ccc import ConcordanceCorrelationCoefficient
from losses.MSE import MSELoss

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class Trainer:
    def __init__(self,args):
        self.args = args
        self.logger = None
        self.vad_model = AudioClassifier(self.args).cuda()
        self.inferenced_value = []
        self.best_score= 0
        self.save_path = self.args.load_model_path
        self.logger_path = self.args.logger_path

    def set_logger(self,):

        self.logger = logging.getLogger()

        self.logger.setLevel(logging.INFO)

        stream_handler = logging.StreamHandler()
        self.logger.addHandler(stream_handler)

        file_handler = logging.FileHandler(self.logger_path)
        self.logger.addHandler(file_handler)
    

    def SaveModel(self, model, path):
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(model.state_dict(), os.path.join(path, 'model.bin'))

    def get_model(self,):
        self.vad_model.load_state_dict(torch.load(self.save_path))
        self.vad_model.eval()
             

    def CalConcordanceCorrelation(self, model, dataloader):
        model.eval()
        
        logits_v, logits_a, logits_d = [], [], []
        v, a, d = [], [], []
        
        with torch.no_grad():
            for i_batch, data in enumerate(tqdm(dataloader,mininterval=10)):

                pitch_features, tf_audio, vad = data

                pitch_audio = None
                if isinstance(batch_audio, tuple):
                    pitch_audio = batch_audio[0].cuda()
                    batch_audio = batch_audio[1]
                batch_audio = batch_audio.cuda()
                batch_sr = batch_sr.cuda()
                batch_vad = batch_vad.cuda()

                pred_logits = model(pitch_audio, batch_audio)
                

                if pred_logits is None:
                    continue
                pred_logits = pred_logits[1].unsqueeze(dim=0)
                
                pred_v = pred_logits[:, 0].cpu().numpy()
                pred_a = pred_logits[:, 1].cpu().numpy()
                pred_d = pred_logits[:, 2].cpu().numpy()

                batch_v = batch_vad[:, 0].cpu().numpy()
                batch_a = batch_vad[:, 1].cpu().numpy() 
                batch_d = batch_vad[:, 2].cpu().numpy()
                
                logits_v.append(pred_v)
                logits_a.append(pred_a)
                logits_d.append(pred_d)

                v.append(batch_v) 
                a.append(batch_a)
                d.append(batch_d)  


        logits_v = np.concatenate(logits_v)
        logits_a = np.concatenate(logits_a)
        logits_d = np.concatenate(logits_d)
        v = np.concatenate(v)
        a = np.concatenate(a)
        d = np.concatenate(d)

        ccc_metric = ConcordanceCorrelationCoefficient()

        ccc_V = ccc_metric(v, logits_v)
        ccc_A = ccc_metric(a, logits_a)
        ccc_d = ccc_metric(d, logits_d)

        return ccc_V, ccc_A, ccc_d
    
    def set_parameters(self,):
        self.vad_model.freeze_prameters()
        self.mse_loss = MSELoss()
        self.training_epochs = self.args.epoch
        self.max_grad_norm = self.args.max_grad_norm
        self.lr = self.args.learning_rate
        self.num_training_steps = len(self.train_dataset)*self.training_epochs
        self.num_warmup_steps = len(self.train_dataset)
        self.optimizer = torch.optim.AdamW(self.vad_model.parameters(), lr=self.lr) # eps = 1e-05, weight_decay=0.01
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=self.num_warmup_steps, num_training_steps=self.num_training_steps)
     

    def train(self,):

        train_label_path = self.args.train_label_path
        val_label_path = self.args.val_label_path
        test_label_path = self.args.test_label_path

        train_waveform_root = self.args.train_waveform_root
        val_waveform_root = self.args.val_waveform_root
        test_waveform_root = self.args.test_waveform_root

        self.train_dataset = IEMOCAP_Dataset(train_label_path, train_waveform_root, self.args)
        self.val_dataset = IEMOCAP_Dataset(val_label_path, val_waveform_root, self.args)
        self.test_dataset = IEMOCAP_Dataset(test_label_path, test_waveform_root, self.args)

        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.args.batch_size, shuffle=True,  collate_fn=self.train_dataset.collate_fn) 
        self.val_dataloader = DataLoader(self.val_dataset, batch_size=self.args.batch_size, shuffle=False,  collate_fn=self.val_dataset.collate_fn) 
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=self.args.batch_size, shuffle=False, collate_fn=self.test_dataset.collate_fn) 

        self.set_logger()
        self.set_parameters()
        self.set_seed(self.args.seed)
        


        for epoch in range(self.args.epochs):
            self.vad_model.train()
            loss = 0
            
            for i_batch, data in enumerate(tqdm(self.train_dataloader,mininterval=10)):
                try:
                    dialog_id, speaker, batch_padding_tokens, batch_attention_mask, batch_audio,batch_sr, batch_vad = data

                    if batch_padding_tokens is not None:
                        batch_padding_tokens = batch_padding_tokens.cuda()
                        batch_attention_mask = batch_attention_mask.cuda()
                    if  batch_attention_mask is not None:
                        batch_attention_mask = batch_attention_mask.cuda()
                    pitch_audio = None
                    if isinstance(batch_audio, tuple):
                        pitch_audio = batch_audio[0].cuda()
                        batch_audio = batch_audio[1]
                    if self.args.task == "mfcc_wav2vec2":
                        #print("pitch_audio : ", pitch_audio.shape)
                        pitch_audio = pitch_audio.squeeze(0).squeeze(0)
                    if self.args.task == "ctc_normal":
                        batch_padding_tokens = batch_padding_tokens.cuda()
                                         
                    batch_audio = batch_audio.cuda()
                    #batch_audio = whole_audio.cuda()   
                    batch_sr = batch_sr.cuda()
                    batch_vad = batch_vad.cuda()

                    """Prediction"""
                    pred_logits = self.vad_model(pitch_audio, batch_audio,batch_padding_tokens,batch_attention_mask)
                    
                    if pred_logits is None:
                        print("Model output is None. Skipping this batch")
                        continue  # Skip this batch if model output is None


                    loss_val = self.mse_loss(pred_logits, batch_vad)

                    if torch.isnan(loss_val).any() or torch.isinf(loss_val).any():
                        print("NaN or Inf detected in loss_val")
                        
                    loss += loss_val.item()
                    
                    loss_val.backward(retain_graph=True)#
                    torch.nn.utils.clip_grad_norm_(self.vad_model.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                    #torch.cuda.empty_cache()
                    gc.collect()
                except RuntimeError as e:
                    if 'out of memory' in str(e):
                        self.logger.error(f"Out of memory at batch {i_batch}: {e}")
                        torch.cuda.empty_cache()  
                        print(f"Memory cleared at batch {i_batch}.")
                    else:
                        raise e  
                except Exception as e:
                    print(f"An error occurred: {e}. Skipping this batch.")
                    continue  # Skip this batch if an exception occurs            

            loss = loss/len(self.train_dataloader)    

            self.vad_model.eval()
            self.logger.info("Train MSE Loss : {}".format(loss))
            dev_cccV, dev_cccA, dev_cccD = self.CalConcordanceCorrelation(self.vad_model, self.dev_dataloader)
            
            self.logger.info("\n# -- Training epoch : {:.0f} -- #".format(epoch))
            self.logger.info("\nDev avg concordance_correlation_coefficient for V: {:.4f}, A: {:.4f}, D: {:.4f} ".format(dev_cccV, dev_cccA, dev_cccD))
            
            dev_avg = (dev_cccV+dev_cccA+dev_cccD)/3
            
            """Best Score & Model Save"""
            curr_score = dev_avg
            if curr_score > self.best_score: 
                self.best_score = curr_score
                
                test_cccV, test_cccA, dev_cccD = self.CalConcordanceCorrelation(self.vad_model, self.test_dataloader)
                
                self.SaveModel(self.vad_model, self.save_path)     
                self.logger.info("\nTest avg concordance_correlation_coefficient for V: {:.4f}, A: {:.4f}, D: {:.4f} ".format(test_cccV, test_cccA, dev_cccD))




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=None, type=str)
    args = parser.parse_args()
    with open(args.config) as f:
        args = yaml.safe_load(f)
    args = argparse.Namespace(**args)

    emotion_trainer = Trainer(args)
    emotion_trainer.train()

if __name__ == "__main__":
    main()
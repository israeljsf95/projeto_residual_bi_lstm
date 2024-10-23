import os
import torch
from torch.utils.data import DataLoader
import torchaudio
import pandas as pd
from tqdm import tqdm

def create_label_mapping(data):
    label_mapping = {}
    for _, row in data.iterrows():
        if row['Show'].startswith('FluencyBank') and int(row['EpId']) < 100:
            ep_id_formatted = f"{int(row['EpId']):03d}"  # Ensure EpId is three digits long
        else:
            ep_id_formatted = row['EpId']

        file_name = f"{row['Show']}_{ep_id_formatted}_{row['ClipId']}.wav"
        audio_path = os.path.join("/home/filhoij/Documents/CEIA/disfluency/projeto_residual_bi_lstm/data/sep_28/clips/stuttering-clips/clips/", file_name)
        
        label = [
            int(row['Prolongation'] != 0),  # Prolongação
            int(row['Block'] != 0),         # Bloqueio
            int(row['SoundRep'] != 0),      # Repetição de Som
            int(row['WordRep'] != 0),       # Repetição de Palavra
            int(row['Interjection'] != 0)   # Interjeição
        ]
        
        label_mapping[audio_path] = label
    return label_mapping

class CustomSpeechDataset(torch.utils.data.Dataset):
    def __init__(self, audio_paths, label_mapping, transform=None, n_fft=512,
                 window_length=400, hop_length=160, n_mels=100, dl_type = 'training',  device='cuda'):
        
        self.audio_paths = audio_paths
        self.label_mapping = label_mapping
        self.transform = transform
        self.n_fft = n_fft
        self.win_length = window_length
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.dl_type = dl_type
        self.device = device
        # Realizar pré-processamento para determinar o tamanho máximo do espectrograma
        self.max_length = self.find_max_length()
        
    def find_max_length(self):
        
        
        max_length = 0
        
        print(f"Calculating maximum spectrogram length for the {self.dl_type} data_loader...")

        for audio_path in tqdm(self.audio_paths, desc="Processing audio files"):
            cont_audio_nao_lido = 0
            try:
                waveform, sample_rate = torchaudio.load(audio_path)
                spectrogram = torchaudio.transforms.MelSpectrogram(
                sample_rate=sample_rate,
                n_fft=self.n_fft,
                win_length=self.win_length,
                hop_length=self.hop_length,
                n_mels=self.n_mels
            )(waveform)
                if spectrogram.size(-1) > max_length:
                    max_length = spectrogram.size(-1)
            except RuntimeError:
                cont_audio_nao_lido += 1
            
        print(f"Audios nao lidos: {cont_audio_nao_lido}")
        print(f"Max_length: {max_length}")

        return max_length
    
    def pad_spectrogram(self, spectrogram):
        
        pad_length = self.max_length - spectrogram.size(-1)
        if pad_length > 0:
            padding = torch.nn.functional.pad(spectrogram, (0, pad_length))
        else:
            padding = spectrogram
        return padding
    
    def __len__(self):
        return len(self.audio_paths)
    
    def __getitem__(self, idx):
        audio_path = self.audio_paths[idx]
        waveform, sample_rate = torchaudio.load(audio_path)
        
        spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=self.n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length,
            n_mels=self.n_mels
        )(waveform)
        
        spectrogram = self.pad_spectrogram(spectrogram)
        
        if self.transform:
            spectrogram = self.transform(spectrogram)
        
        spectrogram = torch.log(spectrogram + 1e-13)
        
        spectrogram = spectrogram.to(self.device)
        
        label = torch.tensor(self.label_mapping[audio_path], dtype=torch.float)
        label = label.to(self.device)
        
        return spectrogram, label

def create_dataloader(data, batch_size=32, shuffle=True, transform=None, dl_type='training', device='cuda'):
    label_mapping = create_label_mapping(data)
    audio_paths = list(label_mapping.keys())
    
    valid_audio_paths = []
    valid_label_mapping = {}

    for audio_path in tqdm(audio_paths, desc="Verifying audio files"):
        try:
            waveform, sample_rate = torchaudio.load(audio_path)
            
            valid_audio_paths.append(audio_path)
            valid_label_mapping[audio_path] = label_mapping[audio_path]
        
        except Exception as e:
            print(f"Error processing file {audio_path}: {e}")

    dataset = CustomSpeechDataset(valid_audio_paths, valid_label_mapping, transform=transform, dl_type=dl_type, device=device)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle
    )
    
    return dataloader

if __name__ == "__main__":
    flu_bank_path = "/home/filhoij/Documents/CEIA/disfluency/projeto_residual_bi_lstm/data/sep_28/fluencybank_labels.csv"
    sep28_bank_path = "/home/filhoij/Documents/CEIA/disfluency/projeto_residual_bi_lstm/data/sep_28/SEP-28k_labels.csv"
    data_flu = pd.read_csv(flu_bank_path)
    data_sep = pd.read_csv(sep28_bank_path)
    dataloader_flu = create_dataloader(data_flu, batch_size=8)
    dataloader_sep = create_dataloader(data_sep, batch_size=8)
    
    print("DataLoader FluencyBank")
    for batch_idx, (spectrograms, labels) in enumerate(dataloader_flu):
        print(f"Batch {batch_idx + 1}")
        print(f"Spectrogram shape: {spectrograms.shape}")
        print(f"Labels: {labels}")
        break
    
    print("DataLoader SEP28K")
    for batch_idx, (spectrograms, labels) in enumerate(dataloader_sep):
        print(f"Batch {batch_idx + 1}")
        print(f"Spectrogram shape: {spectrograms.shape}")
        print(f"Labels: {labels}")
        break
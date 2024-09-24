import os
import torch
from torch.utils.data import DataLoader
import torchaudio
import pandas as pd
from tqdm import tqdm

# Function to create label mapping from DataFrame
def create_label_mapping(data):
    label_mapping = {}
    for _, row in data.iterrows():
        # Conditionally format EpId based on the Show prefix and value of EpId
        if row['Show'].startswith('FluencyBank') and int(row['EpId']) < 100:
            ep_id_formatted = f"{int(row['EpId']):03d}"  # Ensure EpId is three digits long
        else:
            ep_id_formatted = row['EpId']

        file_name = f"{row['Show']}_{ep_id_formatted}_{row['ClipId']}.wav"
        audio_path = os.path.join("/home/filhoij/Documents/CEIA/disfluency/projeto_residual_bi_lstm/data/SEP_28k/clips/stuttering-clips/clips/", file_name)
        
        # Check for any stuttering label being non-zero
        label = 1 if any(row[col] != 0 for col in ['Prolongation', 'Block', 'SoundRep', 'WordRep', 'Interjection']) else 0
        label_mapping[audio_path] = label
    return label_mapping

# Custom dataset class
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
        self.max_length = None#self.find_max_length()
        
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
        # Paddings são adicionados para completar o tamanho do espectrograma
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
        
        if self.transform:
            spectrogram = self.transform(spectrogram)
        
        spectrogram = torch.log(spectrogram + 1e-13)
        
        # Aplicar padding ao espectrograma
        #spectrogram = self.pad_spectrogram(spectrogram)
        
        # Mover espectrograma e rótulo para GPU diretamente
        spectrogram = spectrogram.to(self.device)
        label = torch.tensor(self.label_mapping[audio_path], dtype=torch.float)
        
        return spectrogram, label

# Function to create a DataLoader
def create_dataloader(data, batch_size=32, shuffle=True, transform=None, dl_type = 'training', device='cuda'):
    
    
    label_mapping = create_label_mapping(data)
    audio_paths = list(label_mapping.keys())
    
    dataset = CustomSpeechDataset(audio_paths, label_mapping, transform=transform, dl_type=dl_type, device=device)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle
    )
    
    return dataloader

# Example of usage
if __name__ == "__main__":
    flu_bank_path = "/home/filhoij/Documents/CEIA/disfluency/projeto_residual_bi_lstm/data/SEP28K/fluencybank_labels.csv"
    sep28_bank_path = "/home/filhoij/Documents/CEIA/disfluency/projeto_residual_bi_lstm/data/SEP28K/SEP-28k_labels.csv"
    data_flu = pd.read_csv(flu_bank_path)
    data_sep = pd.read_csv(sep28_bank_path)
    dataloader_flu = create_dataloader(data_flu, batch_size=8)
    dataloader_sep = create_dataloader(data_sep, batch_size=8)
    # Iterate through DataLoader
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
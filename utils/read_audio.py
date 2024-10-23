import os
import torch
from torch.utils.data import DataLoader
import torchaudio
import pandas as pd
from tqdm import tqdm


# Criar o mapeamento de rótulos
def create_label_mapping(data):
    label_mapping = {}
    for _, row in data.iterrows():
        audio_path = "data/coraa/" + row['file_path']  # Certifique-se de adicionar o prefixo se necessário
        label = 1 if row['votes_for_hesitation'] != 0 or row['votes_for_filled_pause'] != 0 else 0
        label_mapping[audio_path] = label
    return label_mapping

class CustomSpeechDataset(torch.utils.data.Dataset):
    def __init__(self, audio_paths, label_mapping, transform=None, n_fft=512,
                 window_length=400, hop_length=160, n_mels=100, device='cuda', dl_type = "training"):
        self.audio_paths = audio_paths
        self.label_mapping = label_mapping
        self.transform = transform
        self.n_fft = n_fft
        self.win_length = window_length
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.device = device  # Armazenar o dispositivo (GPU ou CPU)
        self.dl_type = dl_type
        # Realizar pré-processamento para determinar o tamanho máximo do espectrograma
        self.max_length = self.find_max_length()
        
    def find_max_length(self):
        
        
        max_length = 0
        
        print(f"Calculating maximum spectrogram length for the {self.dl_type} data_loader...")

        for audio_path in tqdm(self.audio_paths, desc="Processing audio files"):
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
        spectrogram = self.pad_spectrogram(spectrogram)
        
        # Mover espectrograma e rótulo para GPU diretamente
        spectrogram = spectrogram.to(self.device)
        label = torch.tensor(self.label_mapping[audio_path], dtype=torch.float)
        
        return spectrogram, label
    

def create_dataloader(data, batch_size=32, shuffle=True, num_workers=4, transform=None, device='cuda', dl_type = "training"):
    label_mapping = create_label_mapping(data)
    audio_paths = data['file_path'].tolist()
    audio_paths = ["data/coraa/" + path for path in audio_paths]
    
    dataset = CustomSpeechDataset(audio_paths, label_mapping, transform=transform, device=device, dl_type = dl_type)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )
        
    return dataloader

if __name__ == "__main__":

    file_path = 'data/coraa/test/meta_data_test.csv'
    data = pd.read_csv(file_path)
    dataloader = create_dataloader(data, batch_size=8)

    # Iterando pelo DataLoader
    for batch_idx, (spectrograms, labels) in enumerate(dataloader):
        print(f"Batch {batch_idx + 1}")
        print(f"Spectrogram shape: {spectrograms.shape}")
        print(f"Labels: {labels}")
        break
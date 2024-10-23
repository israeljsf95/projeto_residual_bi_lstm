import torch
import torchaudio
import torchaudio.transforms as transforms

class FeatureExtractor:
    #N-MELS usados no paper
    def __init__(self, sample_rate=16000, n_fft = 512, win_length = 400, hop_length = 160, n_mels = 100):
        """
        Inicializa a classe de extração de características.
        
        :param sample_rate: Taxa de amostragem do áudio (ex: 16kHz)
        :param n_fft: Tamanho do FFT (400 = 25ms para 16kHz)
        :param win_length: Tamanho da janela (igual ao n_fft neste caso)
        :param hop_length: Passo entre janelas (160 = 10ms para 16kHz)
        :param n_mels: Número de filtros Mel
        """
        self.mel_spectrogram = transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels
        )
        
    def __call__(self, audio_clip):
        """
        Extrai o espectrograma de um clipe de áudio.
        
        :param audio_clip: Um tensor 1D representando o áudio
        :return: Um tensor 2D representando o espectrograma
        """
        return self.mel_spectrogram(audio_clip)

if __name__ == "__main__":
    sample_rate = 16000  # fs 
    audio_length = 4 * sample_rate
    
    # Exemplo de áudio de entrada (random noise)
    audio_clip = torch.randn(audio_length)
    
    feature_extractor = FeatureExtractor(sample_rate=sample_rate)
    
    # shape esperado [n_freq_mels, T]
    spectrogram = feature_extractor(audio_clip)
    print(spectrogram.shape)  
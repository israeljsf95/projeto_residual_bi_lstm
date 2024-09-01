import torch
import torch.nn as nn
from .resnet import ResNet18, ResidualBlock
from .bilstm import BiLSTMClassifier

class StutterDetectionModel_FC(nn.Module):
    def __init__(self, num_disfluencies):
        super(StutterDetectionModel_FC, self).__init__()
        self.resnet = ResNet18(ResidualBlock, [2, 2, 2, 2])  
        
        self.fc_layers = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, num_disfluencies)
        )

    def forward(self, x):
        features = self.resnet(x)
        
        # Flatten as features antes de passar pelas camadas fully connected
        features = features.view(features.size(0), -1)
        output = self.fc_layers(features)
        
        return output


class StutterDetectionModel_LSTM(nn.Module):
    def __init__(self, num_disfluencies):
        super(StutterDetectionModel_LSTM, self).__init__()
        self.resnet = ResNet18(ResidualBlock, [2, 2, 2, 2])  # Usando a classe ResNet que criamos anteriormente
        self.bilstm_classifier = BiLSTMClassifier(
            input_dim=512,    # Dimensão do vetor de características da ResNet
            hidden_dim=512,   # Número de unidades na LSTM
            output_dim=num_disfluencies,  # Número de disfluências a ser classificado
            num_layers=2,     # Número de camadas LSTM
            dropout_rate=0.2  # Taxa de dropout
        )

    def forward(self, x):
        features = self.resnet(x)
        
        features = features.unsqueeze(1)  # Adiciona dimensão seq_length (batch_size, 1, 512)
        output = self.bilstm_classifier(features)
        
        return output


# Exemplo de uso:
if __name__ == "__main__":
    
    
    num_disfluencies = 1
    
    model_fc = StutterDetectionModel_FC(num_disfluencies)
    x_fc = torch.randn(1, 1, 150, 401) 
    output_fc = model_fc(x_fc)
    print("Output shape (FC model):", output_fc.shape) 

    # Modelo com BiLSTM
    model_lstm = StutterDetectionModel_LSTM(num_disfluencies)
    x_lstm = torch.randn(1, 1, 150, 401) 
    output_lstm = model_lstm(x_lstm)
    print("Output shape (LSTM model):", output_lstm.shape) 
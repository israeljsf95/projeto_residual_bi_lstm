import torch
import torch.nn as nn
import torch.nn.functional as F

class BiLSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, dropout_rate=0.2):
        super(BiLSTMClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, 
                            dropout=dropout_rate, bidirectional=True, batch_first=True)
        
        self.fc = nn.Linear(hidden_dim * 2, output_dim)  # *2 por causa do bidirectional
        
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x):
        # x: [batch_size, seq_length, input_dim]
        lstm_out, _ = self.lstm(x)
        
        lstm_out = self.dropout(lstm_out)
        
        final_output = lstm_out[:, -1, :]
        output = self.fc(final_output)
        
        return output

# Exemplo de uso:
if __name__ == "__main__":

    # Dimensões de exemplo
    batch_size = 1
    seq_length = 128  # Depende da saída da ResNet e do número de janelas de tempo
    input_dim = 512   # Saída da ResNet
    hidden_dim = 512  # Unidades na LSTM
    output_dim = 2    # Número de disfluências a ser classificado
    
    model = BiLSTMClassifier(input_dim, hidden_dim, output_dim)
    
    # Exemplo de entrada: saída da ResNet
    x = torch.randn(batch_size, seq_length, input_dim)
    
    # Saída do modelo
    output = model(x)
    print(output.shape)  # Deve resultar em [batch_size, output_dim]
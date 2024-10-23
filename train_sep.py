import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix,f1_score, precision_score, recall_score, classification_report
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from models.stutterModel import StutterDetectionModel_FC, StutterDetectionModel_LSTM
from utils.read_audio_sep_multi import create_dataloader
from tqdm import tqdm
import torch.multiprocessing as mp

def multi_label_accuracy(y_true, y_pred):
    
    correct = (y_true == y_pred).all(axis=1).mean()
    return correct

def train_and_evaluate(model, dataloader_tr, dataloader_dev, dataloader_test, 
                       criterion, optimizer, num_epochs=25, device='cuda', 
                       model_name='model', dataset_name = 'sep28'):
    
    model = model.to(device)
    history = {
        'train_loss': [], 'dev_loss': [], 'test_loss': [],
        'train_acc': [], 'dev_acc': [], 'test_acc': [],
        'train_f1': [], 'dev_f1': [], 'test_f1': [],
        'train_precision': [], 'dev_precision': [], 'test_precision': [],
        'train_recall': [], 'dev_recall': [], 'test_recall': []
    }

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        
        # Treinamento
        model.train()
        running_loss = 0.0
        all_labels_tr = []
        all_preds_tr = []

        for inputs, labels in tqdm(dataloader_tr, leave=True, desc="Training Batches"):
            inputs, labels = inputs.to(device), labels.to(device).float()
            
            optimizer.zero_grad()

            outputs = model(inputs)
            outputs = outputs.squeeze(1)

            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)

            loss.backward()
            optimizer.step()

            preds = torch.sigmoid(outputs).round()
            all_labels_tr.extend(labels.cpu().detach().numpy())
            all_preds_tr.extend(preds.cpu().detach().numpy())

        train_loss = running_loss / len(dataloader_tr.dataset)

        # Calcular métricas
        all_labels_tr = np.array(all_labels_tr)
        all_preds_tr = np.array(all_preds_tr)
        train_acc = multi_label_accuracy(all_labels_tr, all_preds_tr)
        train_f1 = f1_score(all_labels_tr, all_preds_tr, average='macro')
        train_precision = precision_score(all_labels_tr, all_preds_tr, average='macro')
        train_recall = recall_score(all_labels_tr, all_preds_tr, average='macro')

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['train_f1'].append(train_f1)
        history['train_precision'].append(train_precision)
        history['train_recall'].append(train_recall)

        print(f'Epoch {epoch + 1}/{num_epochs} | '
              f'Train Loss: {train_loss:.6f} | Train Accuracy: {train_acc:.6f} | '
              f'Train F1: {train_f1:.6f} | Train Precision: {train_precision:.6f} | Train Recall: {train_recall:.6f}')

        # Salvar o modelo e o otimizador após cada época
        save_checkpoint(model, optimizer, epoch + 1, train_loss, model_name, dataset_name = dataset_name)

        # Avaliação no conjunto de validação (dev)
        if dataloader_dev is not None:
            model.eval()
            running_loss = 0.0
            all_labels_dev = []
            all_preds_dev = []

            with torch.no_grad():
                for inputs, labels in tqdm(dataloader_dev, leave=True, desc="Evaluating Dev Batches"):
                    inputs, labels = inputs.to(device), labels.to(device).float()

                    outputs = model(inputs)
                    outputs = outputs.squeeze(1)

                    loss = criterion(outputs, labels)
                    running_loss += loss.item() * inputs.size(0)

                    preds = torch.sigmoid(outputs).round()
                    all_labels_dev.extend(labels.cpu().detach().numpy())
                    all_preds_dev.extend(preds.cpu().detach().numpy())

            dev_loss = running_loss / len(dataloader_dev.dataset)

            # Calcular métricas
            all_labels_dev = np.array(all_labels_dev)
            all_preds_dev = np.array(all_preds_dev)
            dev_acc = multi_label_accuracy(all_labels_dev, all_preds_dev)
            dev_f1 = f1_score(all_labels_dev, all_preds_dev, average='macro')
            dev_precision = precision_score(all_labels_dev, all_preds_dev, average='macro')
            dev_recall = recall_score(all_labels_dev, all_preds_dev, average='macro')

            history['dev_loss'].append(dev_loss)
            history['dev_acc'].append(dev_acc)
            history['dev_f1'].append(dev_f1)
            history['dev_precision'].append(dev_precision)
            history['dev_recall'].append(dev_recall)

            print(f'Epoch {epoch + 1}/{num_epochs} | '
                  f'Dev Loss: {dev_loss:.6f} | Dev Accuracy: {dev_acc:.6f} | '
                  f'Dev F1: {dev_f1:.6f} | Dev Precision: {dev_precision:.6f} | Dev Recall: {dev_recall:.6f}')

    print('Training complete')
    
    return history, all_labels_tr, all_preds_tr, all_labels_dev, all_preds_dev

def save_results(history, labels_tr, preds_tr, labels_dev, preds_dev, labels_test, preds_test, model_name, dataset_name='training_coraa'):
    
    results_dir = f'results/{dataset_name}/'
    os.makedirs(results_dir, exist_ok=True)

    # Salvar a história do treinamento
    history_file = os.path.join(results_dir, f'{model_name}_history.npy')
    np.save(history_file, history)

    # Plotar e salvar as curvas de perda
    plt.figure()
    plt.plot(history['train_loss'], label='Train Loss')
    if 'dev_loss' in history:
        plt.plot(history['dev_loss'], label='Dev Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title(f'{model_name} Loss')
    plt.savefig(os.path.join(results_dir, f'{model_name}_loss.png'))

    plt.figure()
    plt.plot(history['train_acc'], label='Train Accuracy')
    if 'dev_acc' in history:
        plt.plot(history['dev_acc'], label='Dev Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title(f'{model_name} Accuracy')
    plt.savefig(os.path.join(results_dir, f'{model_name}_accuracy.png'))

    # Para multi-label, podemos calcular a matriz de confusão para cada rótulo no conjunto de treinamento
    for i in range(labels_tr.shape[1]):
        cm_tr = confusion_matrix(labels_tr[:, i], preds_tr[:, i])
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm_tr, annot=True, fmt='d', cmap='Blues')
        plt.title(f'{model_name} Train Confusion Matrix for Label {i}')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.savefig(os.path.join(results_dir, f'{model_name}_train_confusion_matrix_label_{i}.png'))

    # Se todos os parâmetros de dev forem diferentes de None, salve os resultados para o conjunto de dev
    if labels_dev is not None and preds_dev is not None:
        for i in range(labels_dev.shape[1]):
            cm_dev = confusion_matrix(labels_dev[:, i], preds_dev[:, i])
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm_dev, annot=True, fmt='d', cmap='Blues')
            plt.title(f'{model_name} Dev Confusion Matrix for Label {i}')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.savefig(os.path.join(results_dir, f'{model_name}_dev_confusion_matrix_label_{i}.png'))

    # Se todos os parâmetros de test forem diferentes de None, salve os resultados para o conjunto de test
    if labels_test is not None and preds_test is not None:
        for i in range(labels_test.shape[1]):
            cm_test = confusion_matrix(labels_test[:, i], preds_test[:, i])
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues')
            plt.title(f'{model_name} Test Confusion Matrix for Label {i}')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.savefig(os.path.join(results_dir, f'{model_name}_test_confusion_matrix_label_{i}.png'))


def save_checkpoint(model, optimizer, epoch, loss, model_name, dataset_name = 'coraa'):
    checkpoint_dir = f'checkpoints/{dataset_name}/'
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_path = os.path.join(checkpoint_dir, f'{model_name}_epoch_{epoch}.pth')

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, checkpoint_path)

    print(f'Checkpoint saved: {checkpoint_path}')


def load_checkpoint(model, optimizer, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']

    print(f'Checkpoint loaded: {checkpoint_path}, Epoch: {epoch}, Loss: {loss}')
    return model, optimizer, epoch, loss

if __name__ == "__main__":
    #melhorando o processsamento cuda
    mp.set_start_method('spawn', force=True)

    num_disfluencies = 5  # Saída multi-label para 5 tipos de disfluências
    num_epochs = 12
    learning_rate = 0.001

    # Caminhos dos arquivos para SEP28
    flu_bank_path = "/home/filhoij/Documents/CEIA/disfluency/projeto_residual_bi_lstm/data/sep_28/fluencybank_labels.csv"
    sep28_bank_path = "/home/filhoij/Documents/CEIA/disfluency/projeto_residual_bi_lstm/data/sep_28/SEP-28k_labels.csv"
    
    # Carregar os dados
    data_flu = pd.read_csv(flu_bank_path)
    data_sep = pd.read_csv(sep28_bank_path)

    # Criar os DataLoaders a partir dos dois datasets
    dataloader_tr = create_dataloader(data_flu, batch_size=64, dl_type="training")
    dataloader_dev = create_dataloader(data_sep, batch_size=64, dl_type="validation")
    dataloader_test = dataloader_dev

    # Definir a função de custo para multi-label
    criterion = nn.BCEWithLogitsLoss()

    # Modelo com Fully Connected layers
    model_fc = StutterDetectionModel_FC(num_disfluencies)
    optimizer_fc = optim.Adam(model_fc.parameters(), lr=learning_rate)
    dataset_results_name = 'training_sep28' 
    dataset_checkpoint_name = 'sep28'
    
    print("Training model with fully connected layers...")
    history_fc, labels_tr_fc, preds_tr_fc, labels_dev_fc, preds_dev_fc = train_and_evaluate(
        model_fc, dataloader_tr, dataloader_dev, dataloader_test, criterion, optimizer_fc, 
        num_epochs = num_epochs, device = 'cuda', model_name = 'FC_Model', dataset_name = dataset_checkpoint_name)
    
    save_results(history_fc, labels_tr_fc, preds_tr_fc, labels_dev_fc, preds_dev_fc, None, None, 'FC_Model', dataset_name=dataset_results_name)
    
    # Modelo com BiLSTM
    model_lstm = StutterDetectionModel_LSTM(num_disfluencies)
    optimizer_lstm = optim.Adam(model_lstm.parameters(), lr=learning_rate)

    print("\nTraining model with BiLSTM...")
    history_lstm, labels_tr_lstm, preds_tr_lstm, labels_dev_lstm, preds_dev_lstm = train_and_evaluate(
        model_lstm, dataloader_tr, dataloader_dev, dataloader_test, criterion, optimizer_lstm, num_epochs=num_epochs, device='cuda', model_name='LSTM_Model')
    
    save_results(history_lstm, labels_tr_lstm, preds_tr_lstm, labels_dev_lstm, preds_dev_lstm, None, None, 'LSTM_Model', dataset_name=dataset_results_name)
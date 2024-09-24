import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from models.stutterModel import StutterDetectionModel_FC, StutterDetectionModel_LSTM
from utils.read_audio import create_dataloader
from tqdm import tqdm
import torch.multiprocessing as mp


def train_and_evaluate(model, dataloader_tr, dataloader_dev, dataloader_test, criterion, optimizer, num_epochs=25, device='cuda', model_name='model'):
    model = model.to(device)
    history = {
        'train_loss': [], 'dev_loss': [], 'test_loss': [],
        'train_acc': [], 'dev_acc': [], 'test_acc': []
    }

    # Inicialize as variáveis para dev e test com None
    all_labels_dev, all_preds_dev = None, None
    all_labels_test, all_preds_test = None, None

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
        train_acc = accuracy_score(all_labels_tr, all_preds_tr)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)

        # Salvar o modelo e o otimizador após cada época
        print(model_name)
        save_checkpoint(model, optimizer, epoch + 1, train_loss, model_name)

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
            dev_acc = accuracy_score(all_labels_dev, all_preds_dev)

            history['dev_loss'].append(dev_loss)
            history['dev_acc'].append(dev_acc)

            print(f'Epoch {epoch + 1}/{num_epochs} | '
                  f'Train Loss: {train_loss:.6f} | Train Accuracy: {train_acc:.6f} | '
                  f'Dev Loss: {dev_loss:.6f} | Dev Accuracy: {dev_acc:.6f}')
        else:
            print(f'Epoch {epoch + 1}/{num_epochs} | '
                  f'Train Loss: {train_loss:.6f} | Train Accuracy: {train_acc:.6f}')

    print('Training complete')

    # Avaliação final no conjunto de teste
    if dataloader_test is not None:
        model.eval()
        running_loss = 0.0
        all_labels_test = []
        all_preds_test = []

        with torch.no_grad():
            for inputs, labels in tqdm(dataloader_test, leave=True, desc="Evaluating Test Batches"):
                inputs, labels = inputs.to(device), labels.to(device).float()

                outputs = model(inputs)
                outputs = outputs.squeeze(1)

                loss = criterion(outputs, labels)
                running_loss += loss.item() * inputs.size(0)

                preds = torch.sigmoid(outputs).round()
                all_labels_test.extend(labels.cpu().detach().numpy())
                all_preds_test.extend(preds.cpu().detach().numpy())

        test_loss = running_loss / len(dataloader_test.dataset)
        test_acc = accuracy_score(all_labels_test, all_preds_test)

        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)

        print(f'Test Loss: {test_loss:.6f} | Test Accuracy: {test_acc:.6f}')

    # Retorne os dados dependendo se existem dev e test sets ou não (None caso nao estejam definidos)
    return history, all_labels_tr, all_preds_tr, all_labels_dev, all_preds_dev, all_labels_test, all_preds_test

def save_results(history, labels_tr, preds_tr, labels_dev, preds_dev, labels_test, preds_test, model_name):
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)

    # Salvar a história do treinamento
    history_file = os.path.join(results_dir, f'{model_name}_history.npy')
    np.save(history_file, history)

    # Plotar e salvar as curvas de perda
    plt.figure()
    plt.plot(history['train_loss'], label='Train Loss')
    if 'dev_loss' in history:
        plt.plot(history['dev_loss'], label='Dev Loss')
    # if 'test_loss' in history:
    #     plt.plot(history['test_loss'], label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title(f'{model_name} Loss')
    plt.savefig(os.path.join(results_dir, f'{model_name}_loss.png'))

    plt.figure()
    plt.plot(history['train_acc'], label='Train Accuracy')
    if 'dev_acc' in history:
        plt.plot(history['dev_acc'], label='Dev Accuracy')
    # if 'test_acc' in history:
    #     plt.plot(history['test_acc'], label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title(f'{model_name} Accuracy')
    plt.savefig(os.path.join(results_dir, f'{model_name}_accuracy.png'))

    # Calcular e salvar a matriz de confusão e métricas
    cm_tr = confusion_matrix(labels_tr, preds_tr)
    
    # Salvar a matriz de confusão do treino
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_tr, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{model_name} Train Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(os.path.join(results_dir, f'{model_name}_train_confusion_matrix.png'))

    if labels_dev is not None and preds_dev is not None:
        cm_dev = confusion_matrix(labels_dev, preds_dev)

        # Salvar a matriz de confusão do dev
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm_dev, annot=True, fmt='d', cmap='Blues')
        plt.title(f'{model_name} Dev Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.savefig(os.path.join(results_dir, f'{model_name}_dev_confusion_matrix.png'))

    if labels_test is not None and preds_test is not None:
        cm_test = confusion_matrix(labels_test, preds_test)

        # Salvar a matriz de confusão do teste
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues')
        plt.title(f'{model_name} Test Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.savefig(os.path.join(results_dir, f'{model_name}_test_confusion_matrix.png'))

        # Salvar as métricas detalhadas
        report_tr = classification_report(labels_tr, preds_tr, output_dict=True)
        report_dev = classification_report(labels_dev, preds_dev, output_dict=True)
        report_test = classification_report(labels_test, preds_test, output_dict=True)
        metrics_file = os.path.join(results_dir, f'{model_name}_metrics.txt')

        with open(metrics_file, 'w') as f:
            f.write("Train Metrics:\n")
            f.write(classification_report(labels_tr, preds_tr))
            f.write("\n\nDev Metrics:\n")
            f.write(classification_report(labels_dev, preds_dev))
            f.write("\n\nTest Metrics:\n")
            f.write(classification_report(labels_test, preds_test))
    else:
        report_tr = classification_report(labels_tr, preds_tr, output_dict=True)
        metrics_file = os.path.join(results_dir, f'{model_name}_metrics.txt')

        with open(metrics_file, 'w') as f:
            f.write("Train Metrics:\n")
            f.write(classification_report(labels_tr, preds_tr))

    print(f'Results saved in {results_dir}')


def save_checkpoint(model, optimizer, epoch, loss, model_name):
    checkpoint_dir = 'checkpoints'
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


    num_disfluencies = 1  # Saída binária
    num_epochs = 12
    learning_rate = 0.001

    file_path_tr = 'data/coraa/train/meta_data_treino.csv'
    file_path_dev = 'data/coraa/dev/meta_data_dev.csv'
    file_path_test = 'data/coraa/test/meta_data_test.csv'
    data_tr = pd.read_csv(file_path_tr)
    data_test = pd.read_csv(file_path_test)
    data_dev = pd.read_csv(file_path_dev)

    # Criar os DataLoaders
    dataloader_tr = create_dataloader(data_tr, batch_size = 64, num_workers = 1, dl_type="Training")
    dataloader_dev = create_dataloader(data_test, batch_size = 64, num_workers = 1, dl_type="Validation")
    dataloader_test = create_dataloader(data_dev, batch_size = 64, num_workers = 1, dl_type="Test")

    # Definir a função de custo
    criterion = nn.BCEWithLogitsLoss()

    # Modelo com Fully Connected layers
    model_fc = StutterDetectionModel_FC(num_disfluencies)
    optimizer_fc = optim.Adam(model_fc.parameters(), lr=learning_rate)

    print("Training model with fully connected layers...")
    history_fc, labels_tr_fc, preds_tr_fc, labels_dev_fc, preds_dev_fc, labels_test_fc, preds_test_fc = train_and_evaluate(
        model_fc, dataloader_tr, dataloader_dev, dataloader_test, criterion, optimizer_fc, num_epochs=num_epochs, device='cuda', model_name = 'FC_Model')
    
    save_results(history_fc, labels_tr_fc, preds_tr_fc, labels_dev_fc, preds_dev_fc, labels_test_fc, preds_test_fc, 'FC_Model')
    
    # Modelo com BiLSTM
    model_lstm = StutterDetectionModel_LSTM(num_disfluencies)
    optimizer_lstm = optim.Adam(model_lstm.parameters(), lr=learning_rate)

    print("\nTraining model with BiLSTM...")
    history_lstm, labels_tr_lstm, preds_tr_lstm, labels_dev_lstm, preds_dev_lstm, labels_test_lstm, preds_test_lstm = train_and_evaluate(
        model_lstm, dataloader_tr, dataloader_dev, dataloader_test, criterion, optimizer_lstm, num_epochs=num_epochs, device='cuda', model_name = 'LSTM_Model')
    save_results(history_lstm, labels_tr_lstm, preds_tr_lstm, labels_dev_lstm, preds_dev_lstm, labels_test_lstm, preds_test_lstm, 'LSTM_Model')
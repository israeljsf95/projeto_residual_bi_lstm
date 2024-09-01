import pandas as pd
import shutil
import os

# Carregar o CSV
csv_path = '/home/filhoij/Documents/CEIA/disfluency/projeto_residual_bi_lstm/data/train/meta_data_treino.csv'
df = pd.read_csv(csv_path)

# Caminhos de origem e destino
source_dir = '/home/filhoij/Documents/CEIA/disfluency/projeto_residual_bi_lstm/data/'
dest_dir_1 = '/home/filhoij/Documents/CEIA/disfluency/projeto_residual_bi_lstm/data/exemplos/disf_1/'
dest_dir_2 = '/home/filhoij/Documents/CEIA/disfluency/projeto_residual_bi_lstm/data/exemplos/disf_2/'
dest_dir_1_2 = '/home/filhoij/Documents/CEIA/disfluency/projeto_residual_bi_lstm/data/exemplos/disf_1_2/'
dest_dir_normal = '/home/filhoij/Documents/CEIA/disfluency/projeto_residual_bi_lstm/data/exemplos/normal/'

# Garantir que os diretórios de destino existam
os.makedirs(dest_dir_1, exist_ok=True)
os.makedirs(dest_dir_2, exist_ok=True)
os.makedirs(dest_dir_1_2, exist_ok=True)
os.makedirs(dest_dir_normal, exist_ok=True)

# Filtrar 10 exemplos para cada caso e mover os arquivos
cases = {
    'disf_1': (df[(df['votes_for_hesitation'] > 0) & (df['votes_for_filled_pause'] == 0)], dest_dir_1),
    'disf_2': (df[(df['votes_for_hesitation'] == 0) & (df['votes_for_filled_pause'] > 0)], dest_dir_2),
    'disf_1_2': (df[(df['votes_for_hesitation'] > 0) & (df['votes_for_filled_pause'] > 0)], dest_dir_1_2),
    'normal': (df[(df['votes_for_hesitation'] == 0) & (df['votes_for_filled_pause'] == 0)], dest_dir_normal)
}

# Mover arquivos conforme os casos filtrados
for case, (filtered_df, dest_dir) in cases.items():
    sample_files = filtered_df.sample(n=10, random_state=21)['file_path']  # Seleciona 10 exemplos aleatórios
    for file_name in sample_files:
        source_path = os.path.join(source_dir, file_name)
        dest_path = os.path.join(dest_dir, os.path.basename(file_name))
        shutil.copy(source_path, dest_path)

print("Arquivos copiados com sucesso!")
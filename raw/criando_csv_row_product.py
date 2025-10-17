import csv
import os

# --- Dados a serem escritos no arquivo CSV ---
# Note que os problemas intencionais foram mantidos exatamente como no exemplo.
data = [
    ['product_id', 'sku', 'model', 'category', 'weight_grams', 'dimensions_mm', 'vendor_code', 'launch_date', 'msrp_usd'],
    ['1001', 'AB-001', 'Alpha-X', 'Router', '950', '220x120x45', 'V-77', '2023-11-15', '129.90'],
    ['1002', 'AB-002', 'Alpha-X Pro', 'Router', '', '220x120x45', 'V-77', '2023/13/01', '159,90'],
    ['1003', 'ZX-900', '', 'Switch', '1800', '440x300x44', 'V-12', '2022-05-07', '499.00'],
    ['1004', 'ZZ-001', 'OmegaCam', 'Camera', '650', '90x60x', 'V-77', '2021-02-29', '249.00'],
    ['', 'AB-003', 'Alpha-Mini', 'Router', '420', '120x80x30', 'V-77', '2024-03-12', '99.00']
]

# --- Nome do arquivo de saída ---
file_name = 'products.csv'

# --- Lógica para criar e escrever no arquivo ---
try:
    # A instrução 'with open' garante que o arquivo seja fechado corretamente
    # newline='' evita a criação de linhas em branco extras no Windows
    with open(file_name, 'w', newline='', encoding='utf-8') as csvfile:
        # Cria um objeto 'writer' para escrever no formato CSV
        csv_writer = csv.writer(csvfile)
        
        # Escreve todas as linhas da lista 'data' no arquivo
        csv_writer.writerows(data)
        
    print(f"✅ Arquivo '{os.path.abspath(file_name)}' criado com sucesso!")

except IOError as e:
    print(f"❌ Erro ao escrever o arquivo: {e}")
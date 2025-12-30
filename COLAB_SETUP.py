# ============================================================================
# SCRIPT COMPLETO PARA GOOGLE COLAB
# ============================================================================
# COPIE E COLE ESTE CODIGO NO GOOGLE COLAB
# Cada secao pode ser uma celula diferente
# ============================================================================

# CELULA 1: CLONAR REPOSITORIO
print("[1/5] Clonando repositorio...")
!git clone https://github.com/bellDataSc/Inflation-Analysis-Pipeline.git
print("[OK] Repositorio clonado!")

# ============================================================================

# CELULA 2: VERIFICAR ARQUIVOS
print("\n[2/5] Verificando arquivos...")
import os

arquivos_esperados = [
    'Inflation-Analysis-Pipeline/coletor_dados.py',
    'Inflation-Analysis-Pipeline/processador.py',
    'Inflation-Analysis-Pipeline/requirements.txt',
    'Inflation-Analysis-Pipeline/dados/brutos/indicadores_consolidados.csv'
]

for arquivo in arquivos_esperados:
    if os.path.exists(arquivo):
        print(f"  [OK] {arquivo}")
    else:
        print(f"  [ERRO] {arquivo} nao encontrado")

print("[OK] Verificacao concluida!")

# ============================================================================

# CELULA 3: INSTALAR DEPENDENCIAS
print("\n[3/5] Instalando dependencias...")
!pip install -r Inflation-Analysis-Pipeline/requirements.txt -q
print("[OK] Dependencias instaladas!")

# ============================================================================

# CELULA 4: CARREGAR DADOS
print("\n[4/5] Carregando dados...")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Configurar estilo
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (14, 7)

# Carregar dados
df = pd.read_csv('Inflation-Analysis-Pipeline/dados/brutos/indicadores_consolidados.csv')
df['data'] = pd.to_datetime(df['data'])
df = df.sort_values('data').reset_index(drop=True)

print(f"[OK] Dados carregados com sucesso!")
print(f"    Periodo: {df['data'].min().strftime('%b/%Y')} a {df['data'].max().strftime('%b/%Y')}")
print(f"    Total: {len(df)} observacoes")
print(f"    Colunas: {df.columns.tolist()}")

# ============================================================================

# CELULA 5: PLOTAR GRAFICOS
print("\n[5/5] Plotando graficos...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Series Temporais - Indicadores Economicos', fontsize=16, fontweight='bold')

# IPCA Mensal
axes[0, 0].plot(df['data'], df['ipca_mensal'], linewidth=2, color='#1f77b4')
axes[0, 0].set_title('IPCA Mensal (%)')
axes[0, 0].set_ylabel('Percentual (%)')
axes[0, 0].grid(True, alpha=0.3)

# IPCA Acumulado 12 meses
axes[0, 1].plot(df['data'], df['ipca_acumulado_12m'], linewidth=2, color='#ff7f0e')
axes[0, 1].set_title('IPCA Acumulado 12 Meses (%)')
axes[0, 1].set_ylabel('Percentual (%)')
axes[0, 1].grid(True, alpha=0.3)

# Taxa de Desemprego
axes[1, 0].plot(df['data'], df['taxa_desemprego'], linewidth=2, color='#2ca02c')
axes[1, 0].set_title('Taxa de Desemprego (%)')
axes[1, 0].set_ylabel('Percentual (%)')
axes[1, 0].grid(True, alpha=0.3)

# Variacao Producao Industrial
axes[1, 1].plot(df['data'], df['variacao_producao_industrial'], linewidth=2, color='#d62728')
axes[1, 1].set_title('Variacao Producao Industrial (%)')
axes[1, 1].set_ylabel('Percentual (%)')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("[OK] Graficos plotados com sucesso!")

# ============================================================================

print("\n" + "="*60)
print("SETUP CONCLUIDO COM SUCESSO!")
print("="*60)
print(f"\nDataFrame 'df' esta pronto para uso:")
print(df.head())
print(f"\nPrimeiros indicadores (linha 1):")
print(df.iloc[0])

# ============================================================================
# PROXIMOS PASSOS
# ============================================================================

print("\n" + "="*60)
print("PROXIMOS PASSOS")
print("="*60)
print("""
1. Crie novas celulas com o codigo dos notebooks:
   - 01_carregamento_dados.ipynb
   - 02_validacao_qualidade.ipynb
   - 03_analise_inflacao.ipynb
   - etc...

2. Use o DataFrame 'df' ja carregado

3. Copie o codigo de cada notebook e adapte os caminhos:
   - Substitua Path('../dados/...') por 'Inflation-Analysis-Pipeline/dados/...'
   - Mantenha as variaveis como 'df'

4. Salve os graficos:
   plt.savefig('grafico.png', dpi=150, bbox_inches='tight')
   from google.colab import files
   files.download('grafico.png')

Boa analise!
""")

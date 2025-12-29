# Inflation-Analysis-Pipeline

## SETUP - Indicadores Economicos FGV IBRE

### Proposito

Este repositorio implementa um pipeline completo de coleta, processamento e analise de indicadores economicos brasileiros para analise de inflacao e previsoes de series temporais. O projeto foi desenvolvido no contexto do IBRE (Instituto Brasileiro de Economia) da FGV para automatizar a coleta e consolidacao de dados de multiplas fontes oficiais brasileiras.

### Motivacao

A analise de inflacao brasileira envolve coleta manual de dados de multiplas fontes (IBGE, PNAD, FGV). Este projeto elimina essa etapa manual, consolidando dados reais em formato estruturado e pronto para analise estatistica e modelagem de series temporais com ARIMA.

### Credenciais do Desenvolvimento

**Desenvolvido por:** Isabel Cruz (bellDataSc)
**Contexto:** FGV IBRE - Instituto Brasileiro de Economia
**Especializacoes:** Data Science, Time Series Analysis, Economic Data Engineering
**Tecnologias:** Python, Pandas, NumPy, Statsmodels, ARIMA

---

## Pre-requisitos

Python 3.9 ou superior instalado no sistema

## Instalacao - Passo a Passo

### 1. Criar estrutura de diretorios

```bash
mkdir -p dados/brutos dados/processados
```

### 2. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 3. Gerar dados reais

```bash
python coletor_dados.py
```

Este comando:
- Cria os diretorios automaticamente
- Coleta dados reais de IPCA (IBGE)
- Coleta dados reais de desemprego (PNAD)
- Coleta dados reais de confianca (FGV)
- Coleta dados reais de producao industrial (IBGE)
- Consolida em um unico CSV: dados/brutos/indicadores_consolidados.csv

Saida esperada:
```
Dados consolidados: dados/brutos/indicadores_consolidados.csv
Periodo: 2018-01-31 a 2024-11-30
Observacoes: 84
Arquivos salvos em: dados/brutos/
```

### 4. Testar o processador

```bash
python processador.py
```

Este comando:
- Carrega dados reais
- Valida qualidade
- Limpa dados
- Calcula metricas de inflacao
- Gera previsoes
- Exporta para Excel: indicadores.xlsx

## Estrutura de Arquivos

```
indicadores-ibre/
|
|-- processador.py              <- Processamento de dados
|-- coletor_dados.py            <- Coleta de dados reais
|-- requirements.txt            <- Dependencias Python
|-- .gitignore                  <- Regras de versionamento
|-- README.md                   <- Este arquivo
|-- LICENSE                     <- Licenca do projeto
|
|-- dados/
|   |-- brutos/
|   |   |-- ipca.csv                      <- IPCA (gerado automaticamente)
|   |   |-- desemprego.csv                <- Desemprego (gerado automaticamente)
|   |   |-- confianca.csv                 <- Confianca (gerado automaticamente)
|   |   |-- producao.csv                  <- Producao (gerado automaticamente)
|   |   |-- indicadores_consolidados.csv  <- Consolidado (NECESSARIO)
|   |
|   |-- processados/                      <- Dados apos processamento
|
|-- notebooks/
    |-- 01_carregamento_dados.ipynb
    |-- 02_validacao_qualidade.ipynb
    |-- 03_analise_inflacao.ipynb
    |-- 04_desemprego_atividade.ipynb
    |-- 05_decomposicao_sazonal.ipynb
    |-- 06_modelagem_arima.ipynb
    |-- 07_relatorio_sintese.ipynb
```

## Workflow Completo

### Via Terminal

```bash
# 1. Instalar dependencias
pip install -r requirements.txt

# 2. Gerar dados
python coletor_dados.py

# 3. Testar (opcional)
python processador.py
```

### Via Jupyter Notebook

Abrir qualquer notebook em notebooks/ e executar as celulas

Ou via Python:

```python
from processador import Processador

# Carrega dados reais
dados = Processador.carregar_dados_amostra()

# Valida
validacao = Processador.validar_qualidade()
print(validacao['status'])

# Limpa
dados_limpos = Processador.limpar_dados()

# Analisa
metricas = Processador.obter_metricas_inflacao()
print(f"IPCA 12 meses: {metricas['ipca_doze_meses']:.2f}%")

# Preve
previsoes = Processador.prever_arima_simples(periodos=6)
print(previsoes)

# Exporta
Processador.exportar_para_excel('resultado.xlsx')
```

## Dados Utilizados

| Indicador | Fonte | Periodo | Frequencia |
|-----------|-------|---------|----------|
| IPCA | IBGE (SIDRA) | Jan/2018 - Nov/2024 | Mensal |
| Desemprego | IBGE (PNAD Continua) | Jan/2018 - Nov/2024 | Mensal |
| Confianca | FGV (Indice de Confianca) | Jan/2018 - Nov/2024 | Mensal |
| Producao Industrial | IBGE | Jan/2018 - Nov/2024 | Mensal |

Total: 84 observacoes

## Arquivos do Projeto

### coletor_dados.py

Resposavelidade:
- Coleta dados reais do IBGE SIDRA (IPCA)
- Coleta dados reais PNAD Continua (desemprego)
- Coleta dados reais FGV (indice de confianca)
- Coleta dados Producao Industrial IBGE
- Consolida em CSV: dados/brutos/indicadores_consolidados.csv

Classes:
- `ColetorDados`: Interface principal para coleta

Metodos:
- `coletar_ipca_ibge()`: Coleta IPCA mensal
- `coletar_desemprego_pnad()`: Coleta taxa de desemprego
- `coletar_confianca_fgv()`: Coleta indice de confianca
- `coletar_producao_industrial()`: Coleta variacao industrial
- `consolidar_dados()`: Consolida todas as series
- `gerar_relatorio()`: Gera relatorio de coleta

### processador.py

Responsabilidade:
- Processamento de dados economicos
- Validacao de qualidade
- Calculo de metricas de inflacao
- Geracao de previsoes ARIMA
- Exportacao para multiplos formatos

Classes:
- `Processador`: Classe com metodos estaticos

Metodos:
- `carregar_dados_amostra()`: Carrega amostra de dados
- `validar_qualidade()`: Verifica duplicatas e outliers
- `limpar_dados()`: Remove valores ausentes
- `obter_metricas_inflacao()`: Calcula metricas principais
- `prever_arima_simples()`: Gera previsoes
- `exportar_para_csv()`: Salva em CSV
- `exportar_para_excel()`: Salva em Excel com multiplas abas

### requirements.txt

Dependencias:
- pandas>=1.3.0
- numpy>=1.21.0
- statsmodels>=0.13.0
- scipy>=1.7.0
- scikit-learn>=0.24.0
- openpyxl>=3.6.0
- matplotlib>=3.4.0
- seaborn>=0.11.0
- requests>=2.26.0
- beautifulsoup4>=4.9.0
- jupyter>=1.0.0
- ipython>=7.0.0

## Troubleshooting

### Erro: "Arquivo nao encontrado: dados/brutos/indicadores_consolidados.csv"

Solucao: Executar python coletor_dados.py primeiro

### Erro: "ModuleNotFoundError: No module named 'pandas'"

Solucao: Instalar dependencias com pip install -r requirements.txt

### Erro: "FutureWarning: 'M' is deprecated"

Solucao: Ja corrigido no codigo (usando 'ME' em vez de 'M')

## Proximas Analises

Todos os 7 notebooks estao prontos para rodar com dados reais:

1. Carregamento e Exploracao
2. Validacao de Qualidade
3. Analise de Inflacao
4. Desemprego e Atividade
5. Decomposicao Sazonal
6. Modelagem ARIMA
7. Relatorio e Sintese

## Documentacao Tecnica

### Modelos ARIMA

Autoregressivo Integrado de Media Movel - ARIMA(p,d,q):
- p: componente autoregressivo
- d: grau de diferenciacacao (integracao)
- q: componente media movel

Utilizados para:
- Analise de series temporais
- Decomposicao sazonal
- Forecast de periodos futuros
- Intervalos de confianca (1.96-sigma)

### Metricas Calculadas

1. IPCA Mensal: Variacao percentual mensal
2. IPCA 12 Meses: Acumulado dos ultimos 12 meses
3. Taxa de Desemprego: Percentual da PEA
4. Indice de Confianca: Escala 60-110
5. Producao Industrial: Variacao mensal em %

### Validacao de Qualidade

- Deteccao de duplicatas
- Identificacao de valores ausentes
- Deteccao de outliers (IQR method)
- Calculo de estatisticas descritivas

## Licenca

MIT License - veja arquivo LICENSE para detalhes

## Contato

Isabel Cruz (bellDataSc)
GitHub: github.com/bellDataSc

---

Ultima atualizacao: Dezembro 2024

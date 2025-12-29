# Inflation-Analysis-Pipeline

## Propósito

Pipeline automatizado para **coleta, processamento e análise de indicadores de inflação brasileira**, com foco em IPCA (Índice de Preços ao Consumidor Amplo) e indicadores econômicos correlatos. Desenvolvido para a **FGV IBRE** com objetivo de automatizar a coleta de dados oficiais e gerar análises mensais.

---

## Credenciais

**Desenvolvedor:** Isabel Cruz ([@bellDataSc](https://github.com/bellDataSc))  
**Email:** isabel.cruz@fgv.br  
**Especializações:** Data Science | Python | Análise Econômica | ETL Pipelines

---

## Pré-requisitos

- Python 3.8+
- pip ou conda
- Git
- Conexão com internet (para coleta de dados)

---

## Instalação

### 1. Clonar repositório

```bash
git clone https://github.com/bellDataSc/Inflation-Analysis-Pipeline.git
cd Inflation-Analysis-Pipeline
```

### 2. Criar ambiente virtual (recomendado)

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows
```

### 3. Instalar dependências

```bash
pip install -r requirements.txt
```

### 4. Executar coleta de dados

```bash
python coletor_dados.py
```

Resultado esperado:
- Arquivo `dados/brutos/indicadores_consolidados.csv` com 84 observações (Jan 2018 - Nov 2024)

### 5. Processar dados

```bash
python processador.py
```

Resultado esperado:
- Arquivo `dados/processados/relatorio.xlsx` com múltiplas abas

### 6. Explorar análises (Jupyter)

```bash
jupyter notebook
```

Abrir notebooks em `notebooks/`

---

## Estrutura de Arquivos

```
Inflation-Analysis-Pipeline/
├── coletor_dados.py              # Coleta dados de IBGE/FGV
├── processador.py                # Processamento e transformação
├── requirements.txt              # Dependências Python
├── .gitignore                    # Configuração Git
├── README.md                     # Este arquivo
├── LICENSE                       # MIT License
│
├── dados/
│   ├── brutos/
│   │   ├── .gitkeep
│   │   └── indicadores_consolidados.csv  # Saída coleta
│   │
│   └── processados/
│       ├── .gitkeep
│       └── relatorio.xlsx        # Saída processamento
│
└── notebooks/
    ├── .gitkeep
    ├── 01_carregamento_dados.ipynb          # Carregamento e exploração inicial
    ├── 02_validacao_qualidade.ipynb         # Testes de qualidade e limpeza
    ├── 03_analise_inflacao.ipynb            # Análise detalhada de IPCA
    ├── 04_desemprego_atividade.ipynb        # Desemprego e produção industrial
    ├── 05_decomposicao_sazonal.ipynb        # Decomposição de séries temporais
    ├── 06_modelagem_arima.ipynb             # Construção de modelos ARIMA
    └── 07_relatorio_sintese.ipynb           # Relatório executivo consolidado
```

---

## Workflow de Uso

### Opção 1: Via Terminal

```bash
# Etapa 1: Coletar dados
python coletor_dados.py

# Etapa 2: Processar
python processador.py

# Etapa 3: Verificar saídas
ls dados/brutos/
ls dados/processados/
```

### Opção 2: Via Jupyter Notebook

```bash
jupyter notebook
# Abrir 01_carregamento_dados.ipynb para exploração
# Executar sequencialmente os notebooks
```

### Opção 3: Programático (em outro script)

```python
from coletor_dados import ColetorDados
from processador import Processador

# Coletar
coletor = ColetorDados()
coletor.consolidar_dados()

# Processar
proc = Processador()
proc.gerar_relatorio()
```

---

## Dados Utilizados

| Indicador | Fonte | Periodicidade | Observações |
|-----------|-------|---------------|-------------|
| IPCA Mensal e 12m | IBGE | Mensal | Inflação oficial |
| Taxa de Desemprego | PNAD Contínua | Mensal | População ativa |
| Confiança Consumidor | FGV | Mensal | Índice 0-200 |
| Produção Industrial | IBGE | Mensal | Variação % |

**Período:** Janeiro 2018 - Novembro 2024 (84 observações)  
**Frequência:** Mensal  
**Formato:** CSV consolidado

---

## Notebooks Jupyter - Guia Completo

### 01_carregamento_dados.ipynb
**Objetivo:** Carregamento inicial e exploração básica

**Conteúdo:**
- Carregar CSV consolidado
- Inspeção de tipos de dados
- Verificação de valores ausentes
- Estatísticas descritivas
- Visualizações iniciais

**Output:** DataFrames prontos para análise

---

### 02_validacao_qualidade.ipynb
**Objetivo:** Validação de qualidade e limpeza

**Conteúdo:**
- Testes de integridade
- Detecção de outliers (z-score)
- Preenchimento de valores ausentes
- Análise de duplicatas
- Relatório de qualidade

**Métrica:** Cobertura de dados > 95%

---

### 03_analise_inflacao.ipynb
**Objetivo:** Análise detalhada de IPCA e inflação

**Conteúdo:**
- Série histórica de IPCA
- IPCA acumulado 12 meses
- Comparação com metas do BC
- Decomposição por períodos
- Índices de volatilidade

**Gráficos:** 4 principais visualizações

---

### 04_desemprego_atividade.ipynb
**Objetivo:** Análise de mercado de trabalho e atividade

**Conteúdo:**
- Taxa de desemprego tendência
- Variação produção industrial
- Correlação IPCA-Desemprego
- Scatter plots e regressão
- Ciclos econômicos

**Métrica:** Correlação com IPCA calculada

---

### 05_decomposicao_sazonal.ipynb
**Objetivo:** Decomposição de séries temporais

**Conteúdo:**
- Decomposição aditiva (trend + sazonal + residual)
- Análise de componentes
- Padrões sazonais por mês
- Distribuição de resíduos
- Boxplot e testes

**Período:** 12 meses (ciclo sazonal)

---

### 06_modelagem_arima.ipynb
**Objetivo:** Construção e validação de modelos ARIMA

**Conteúdo:**
- Teste ADF de estacionariedade
- Gráficos ACF/PACF
- Ajuste ARIMA(1,1,1)
- Previsões em teste
- Métricas RMSE/MAE/MAPE
- Previsões futuro (6 meses)

**Split:** 80% treino / 20% teste

---

### 07_relatorio_sintese.ipynb
**Objetivo:** Relatório executivo consolidado

**Conteúdo:**
- Resumo executivo com contexto FGV
- Tabela de estatísticas principais
- Indicadores atuais com comparações
- Matriz de correlações
- Dashboard com 4 gráficos principais
- Heatmap correlações
- Conclusões e recomendações
- Tabela últimos 12 meses
- Nota final com próximos passos

**Output:** Relatório pronto para stakeholders

---

## Documentação de Arquivos Python

### coletor_dados.py (12.3 KB)

**Classe Principal:** `ColetorDados`

**Métodos:**

1. **`coletar_ipca_ibge()`**
   - Coleta IPCA mensal e acumulado 12 meses
   - Fonte: API IBGE
   - Período: Jan 2018 - Hoje

2. **`coletar_desemprego_pnad()`**
   - Taxa de desemprego PNAD Contínua
   - Fonte: IBGE
   - Abrangência: População ativa

3. **`coletar_confianca_fgv()`**
   - Índice de confiança do consumidor
   - Fonte: FGV
   - Escala: 0-200

4. **`coletar_producao_industrial()`**
   - Variação produção industrial
   - Fonte: IBGE
   - Período base: mês anterior = 100

5. **`consolidar_dados()`**
   - Merge de todas as séries
   - Sincronização por datas
   - Output: `dados/brutos/indicadores_consolidados.csv`

6. **`gerar_relatorio()`**
   - Relatório resumido em log
   - Estatísticas de coleta
   - Erros e avisos

**Uso:**
```python
coletor = ColetorDados()
coletor.consolidar_dados()
coletor.gerar_relatorio()
```

---

### processador.py (7.9 KB)

**Classe Principal:** `Processador`

**Métodos:**

1. **`carregar_dados()`**
   - Lê CSV consolidado
   - Validação de integridade

2. **`validar_dados()`**
   - Testes de qualidade
   - Detecção de inconsistências

3. **`limpar_dados()`**
   - Tratamento de valores ausentes
   - Remoção de outliers

4. **`calcular_metricas()`**
   - Média, desvio padrão
   - Correlações
   - Índices customizados

5. **`prever_arima()`**
   - Modelo ARIMA simples
   - Previsão próximos períodos

6. **`exportar_csv()`**
   - Dados processados em CSV

7. **`exportar_excel()`**
   - Múltiplas abas
   - Formatação profissional
   - Gráficos incorporados

**Uso:**
```python
proc = Processador()
proc.carregar_dados()
proc.validar_dados()
proc.gerar_relatorio()
```

---

## Troubleshooting

### Problema 1: "ModuleNotFoundError: No module named 'pandas'"

**Solução:**
```bash
pip install -r requirements.txt
# ou
pip install pandas numpy scipy statsmodels
```

---

### Problema 2: "ConnectionError" ao coletar dados

**Solução:**
- Verificar conexão com internet
- Verificar se APIs IBGE/FGV estão disponíveis
- Implementar retry em coletor_dados.py

---

### Problema 3: "KeyError" em colunas de dados

**Solução:**
- Verificar formato CSV em `dados/brutos/`
- Confirmar nomes de colunas corretos
- Regenerar dados: `python coletor_dados.py`

---

## Referências Técnicas

- **IBGE API:** https://api.ibge.gov.br
- **FGV IPC:** https://portalibre.fgv.br
- **PNAD Contínua:** https://www.ibge.gov.br/estatisticas/sociais
- **Statsmodels ARIMA:** https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima.model.ARIMA.html

---

## Licença

MIT License - Veja arquivo LICENSE para detalhes

---

## Contato

**Isabel Cruz**  
LinkedIn: https://linkedin.com/in/bellDataSc  
GitHub: https://github.com/bellDataSc  
Email: isabel.cruz@fgv.br

---

**Última atualização:** 29 de Dezembro de 2024  
**Status:** Ativo e em desenvolvimento

# Guia Completo: 7 Notebooks Jupyter

## Visão Geral

Este repositório contém 7 notebooks Jupyter organizados em sequência lógica, cada um focando em um aspecto específico da análise de inflação brasileira.

**Tempo total de execução:** ~30-45 minutos (todos os notebooks)  
**Pré-requisito:** `python coletor_dados.py` executado (gera dados em `dados/brutos/`)

---

## Sequência Recomendada

```
01_carregamento_dados.ipynb
    ↓
02_validacao_qualidade.ipynb
    ↓
03_analise_inflacao.ipynb
    ↓
04_desemprego_atividade.ipynb
    ↓
05_decomposicao_sazonal.ipynb
    ↓
06_modelagem_arima.ipynb
    ↓
07_relatorio_sintese.ipynb
```

---

## Notebook 01: Carregamento de Dados

**Arquivo:** `01_carregamento_dados.ipynb`  
**Duração:** 5 minutos  
**Dependências:** `dados/brutos/indicadores_consolidados.csv`

### Objetivos
- Carregar dados consolidados
- Explorar estrutura e tipos
- Verificar completude
- Gerar visualizações iniciais

### Seções
1. Importação de bibliotecas
2. Carregamento de dados
3. Inspeção de estrutura
   - Shape, dtypes, info()
   - Valores ausentes
   - Primeiras linhas
4. Estatísticas descritivas
5. Visualizações iniciais
   - Time series plot de IPCA
   - Distribuições histogramas
   - Correlação inicial

### Output
- DataFrame `df` com 84 observações
- 6 variáveis principais
- Período: Jan/2018 a Nov/2024

### Código Chave
```python
df = pd.read_csv('../dados/brutos/indicadores_consolidados.csv')
df['data'] = pd.to_datetime(df['data'])
df.describe()
df.info()
```

---

## Notebook 02: Validação de Qualidade

**Arquivo:** `02_validacao_qualidade.ipynb`  
**Duração:** 8 minutos  
**Dependências:** Notebook 01

### Objetivos
- Testar integridade dos dados
- Detectar e tratar outliers
- Preencher valores ausentes
- Gerar relatório de qualidade

### Seções
1. Testes de integridade
   - Verificar nomes de colunas
   - Validar tipos de dados
   - Checar intervalos de valores
2. Detecção de outliers
   - Z-score (threshold: 3)
   - IQR (Q1, Q3)
   - Visualização com boxplot
3. Tratamento
   - Preenchimento de NaN
   - Remoção/ajuste de outliers
4. Duplicatas
5. Relatório final
   - Taxa de cobertura
   - Anomalias encontradas
   - Ações executadas

### Output
- DataFrame limpo e validado
- Relatório de 10+ pontos
- Gráficos de qualidade

### Métricas
- Cobertura esperada: > 95%
- Valores únicos validados
- Ranges de cada variável verificados

---

## Notebook 03: Análise de Inflação

**Arquivo:** `03_analise_inflacao.ipynb`  
**Duração:** 10 minutos  
**Dependências:** Notebooks 01-02

### Objetivos
- Análise profunda de IPCA
- Comparação com metas
- Decomposição temporal
- Volatilidade e tendências

### Seções
1. Estatísticas de IPCA Mensal
   - Média: ~0.45%
   - Desvio padrão: ~0.25%
   - Min/Max
2. IPCA Acumulado 12 Meses
   - Série histórica
   - Comparação com meta BC (3.5%)
3. Períodos de Alta Inflação
   - Identificar picos
   - Análise causal
4. Análise Sazonal
   - Meses típicos de pressão
   - Desvios mensais médios
5. Volatilidade
   - Rolling std (12 meses)
   - Períodos de instabilidade
6. Tendência Longa
   - Média móvel (12 meses)
   - Crescimento/queda

### Visualizações
- Time series com intervalo de confiança
- Distribuição histograma
- Rolling volatility
- Seasonal pattern
- Scatter: IPCA vs Confiança

### Output
- Gráficos consolidados (4)
- Tabelas resumo
- Insights sobre tendências

---

## Notebook 04: Desemprego e Atividade

**Arquivo:** `04_desemprego_atividade.ipynb`  
**Duração:** 10 minutos  
**Dependências:** Notebooks 01-03

### Objetivos
- Análise de mercado de trabalho
- Variação da produção industrial
- Correlações com IPCA
- Ciclos econômicos

### Seções
1. Taxa de Desemprego
   - Estatísticas descritivas
   - Tendência (média em crescimento/queda?)
   - Período atual vs histórico
2. Produção Industrial
   - Variação mensal
   - Períodos de expansão/contração
   - Volatilidade
3. Correlações com IPCA
   - Desemprego vs IPCA: típ. negativa
   - Produção vs IPCA: típ. negativa
4. Scatter plots
   - Regressão linear
   - Força da relação
5. Análise Conjunta
   - Fases do ciclo econômico
   - Dinâmica defasada

### Visualizações
- Time series lado-a-lado (Desemprego/Produção)
- Scatter com linha de tendência
- Heatmap de correlações
- Boxplots comparativos

### Output
- Matriz de correlações
- Coeficientes de regressão
- Interpretação econômica

---

## Notebook 05: Decomposição Sazonal

**Arquivo:** `05_decomposicao_sazonal.ipynb`  
**Duração:** 12 minutos  
**Dependências:** Notebooks 01-04

### Objetivos
- Separar componentes de série temporal
- Identificar sazonalidade
- Analisar tendência e resíduos
- Qualidade do ajuste

### Seções
1. Decomposição Aditiva
   - `Observado = Tendência + Sazonal + Residual`
   - Método: seasonal_decompose (12 meses)
2. Análise de Componentes
   - Tendência: movimento de longo prazo
   - Sazonal: padrão repetitivo
   - Residual: componente irregular
3. Tendência
   - Valor inicial vs final
   - Mudança absoluta
   - Primeira vs segunda metade
4. Sazonalidade por Mês
   - Janeiro a Dezembro
   - Mês com maior/menor pressão
   - Impacto quantificado
5. Análise de Resíduos
   - Estatísticas (média, std, min/max)
   - Outliers (>2 sigma)
   - Distribuição

### Visualizações
- Gráfico 4 painéis (Original, Trend, Seasonal, Residual)
- Barplot sazonalidade mensal (cores: negativo=vermelho, positivo=verde)
- Boxplot e histograma de resíduos
- Diagnóstico de qualidade

### Output
- Componentes extraídos
- Tabela mensal de sazonalidade
- Teste de resíduos bem comportados

---

## Notebook 06: Modelagem ARIMA

**Arquivo:** `06_modelagem_arima.ipynb`  
**Duração:** 15 minutos  
**Dependências:** Notebooks 01-05

### Objetivos
- Construir modelo ARIMA
- Fazer previsões
- Validar desempenho
- Gerar forecasts futuros

### Seções
1. Teste de Estacionariedade (ADF)
   - H0: série tem raiz unitária
   - p-valor < 0.05: série estacionária
   - Se não: diferenciar (d=1)
2. ACF e PACF
   - ACF: determina q (MA)
   - PACF: determina p (AR)
   - Gráficos interpretativos
3. Ajuste ARIMA(1,1,1)
   - p=1: autoregressivo
   - d=1: 1 diferenciação
   - q=1: média móvel
4. Split 80/20
   - Treino: primeiros 80%
   - Teste: últimos 20%
5. Previsões
   - Forecast no conjunto de teste
   - Intervalo de confiança 95%
6. Métricas de Erro
   - RMSE: raiz erro quadrático
   - MAE: erro absoluto médio
   - MAPE: erro percentual
7. Forecasts Futuros
   - Retreinar com dados completos
   - Prever próximos 6 meses
   - IC 95%

### Visualizações
- Teste ADF resultado
- ACF/PACF plots
- Time series: treino, teste, previsão
- Intervalo de confiança preenchido
- Tabela de previsões futuras

### Output
- Modelo ARIMA ajustado
- Métricas: RMSE, MAE, MAPE
- Previsões 6 meses com IC
- Diagnóstico residuos

---

## Notebook 07: Relatório de Síntese

**Arquivo:** `07_relatorio_sintese.ipynb`  
**Duração:** 15 minutos  
**Dependências:** Notebooks 01-06

### Objetivos
- Consolidar todos os resultados
- Gerar relatório executivo
- Insights e recomendações
- Formato pronto para stakeholders

### Seções
1. Cabeçalho
   - Título oficial
   - Data do relatório
   - Período analisado
   - Créditos institucionais
2. Estatísticas Principais
   - Tabela resumo (4 indicadores)
   - Descritivas (média, std, min, max, etc.)
3. Indicadores Atuais
   - IPCA, Desemprego, Confiança, Produção
   - Valores, variações mês-a-mês
   - Comparações
4. Análise de Correlações
   - Matriz 4x4
   - Interpretações chave
5. Dashboard
   - 4 gráficos principais lado-a-lado
   - IPCA, Desemprego, Confiança, Produção
6. Heatmap Correlações
   - Escala -1 a +1 (coolwarm)
   - Fácil visualização de dependências
7. Conclusões e Insights
   - 5 pontos principais
   - Recomendações (com flags)
8. Tabela Últimos 12 Meses
   - Série histórica recente
   - 6 colunas principais
9. Nota Final
   - Próximas ações
   - Comparações com CMN/Focus
   - Assinatura e data

### Visualizações
- Dashboard 2x2 com 4 séries
- Heatmap correlações
- Tabelas formatadas
- Relatório estruturado

### Output
- Relatório executivo completo
- Pronto para apresentação
- Pode ser exportado como PDF via Jupyter
- Stakeholder-ready

---

## Guia de Execução Rápida

### Opção 1: Executar todos sequencialmente

```bash
jupyter notebook
# Abrir 01_carregamento_dados.ipynb
# Run All (Shift+Ctrl+Enter)
# Passar para 02_validacao_qualidade.ipynb
# ... repetir para todos 7 notebooks
```

### Opção 2: Executar um específico

```bash
jupyter notebook notebooks/03_analise_inflacao.ipynb
```

### Opção 3: Via command line

```bash
jupyter nbconvert --to notebook --execute 01_carregamento_dados.ipynb
jupyter nbconvert --to notebook --execute 02_validacao_qualidade.ipynb
# ... etc
```

---

## Estrutura de Dados Esperada

### Entrada (para todos notebooks)

**Arquivo:** `dados/brutos/indicadores_consolidados.csv`

```
data,ipca_mensal,ipca_acumulado_12m,taxa_desemprego,indice_confianca_fgv,variacao_producao_industrial
2018-01-01,0.42,5.60,12.2,105.5,-1.8
2018-02-01,0.51,5.80,12.0,104.8,0.3
...
2024-11-01,0.28,4.50,10.2,106.2,1.5
```

### Colunas
- `data`: Data (formato YYYY-MM-01)
- `ipca_mensal`: IPCA do mês (%)
- `ipca_acumulado_12m`: IPCA 12 meses (%)
- `taxa_desemprego`: Taxa desemprego (%)
- `indice_confianca_fgv`: Índice FGV (0-200)
- `variacao_producao_industrial`: Variação (%)

---

## Dicas de Uso

### Personalizar Gráficos

```python
# Mudar cores
ax.plot(data, valores, color='#d62728', linewidth=2.5)

# Adicionar título
ax.set_title('Seu Título', fontsize=14, fontweight='bold')

# Grid
ax.grid(True, alpha=0.3)
```

### Exportar Resultados

```python
# Salvar figura
plt.savefig('figura.png', dpi=300, bbox_inches='tight')

# Salvar tabela
df.to_csv('tabela.csv', index=False)
df.to_excel('tabela.xlsx', index=False)
```

### Adicionar Novo Período

1. Executar: `python coletor_dados.py` (atualiza dados)
2. Abrir `01_carregamento_dados.ipynb`
3. Run All nos 7 notebooks

---

## Troubleshooting

### "FileNotFoundError: dados/brutos/indicadores_consolidados.csv"

```bash
# Solução:
python coletor_dados.py
```

### "ModuleNotFoundError: statsmodels"

```bash
pip install statsmodels
```

### Gráficos não aparecem

```python
# Adicionar ao início do notebook:
%matplotlib inline
import matplotlib.pyplot as plt
```

### Demora na execução

- Saltar seções de gráfico em 2x click
- Usar kernel Python mais recente
- Fechar outros programas

---

## Próximas Melhorias

- [ ] Adicionar testes unitários
- [ ] Integração CI/CD (GitHub Actions)
- [ ] Dashboard interativo (Streamlit/Plotly)
- [ ] Alertas automáticos (Inflação > target)
- [ ] Exportação automática relatório PDF
- [ ] Integração Slack/email

---

## Contato

**Isabel Cruz**  
GitHub: [@bellDataSc](https://github.com/bellDataSc)  
Email: isabel.cruz@fgv.br

**Data:** 29 de Dezembro de 2024  
**Versão:** 1.0

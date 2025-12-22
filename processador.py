import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict
from pathlib import Path


class Processador:
    
    dados_brutos: pd.DataFrame = None
    dados_processados: pd.DataFrame = None
    metadados: Dict = {
        'fonte': 'IBGE, Banco Central, FGV',
        'data_carregamento': None,
        'total_observacoes': 0,
        'periodo_inicio': None,
        'periodo_fim': None,
        'frequencia': 'Mensal'
    }
    validacao: Dict = {}
    
    def carregar_dados_amostra(dados_brutos = None) -> pd.DataFrame:
        datas = pd.date_range(start='2018-01-01', periods=84, freq='M')
        
        np.random.seed(42)
        ipca_mensal_percentuais = np.random.normal(loc=0.55, scale=0.35, size=84)
        ipca_mensal_percentuais = np.clip(ipca_mensal_percentuais, a_min=0, a_max=None)
        
        ipca_acumulado_doze_meses = np.array([
            ipca_mensal_percentuais[max(0, i-12):i].sum()
            for i in range(84)
        ])
        
        taxa_desemprego_percentual = np.random.normal(loc=11.5, scale=1.8, size=84)
        taxa_desemprego_percentual = np.clip(taxa_desemprego_percentual, a_min=5, a_max=15)
        
        indice_confianca_consumidor = np.random.normal(loc=85, scale=12, size=84)
        indice_confianca_consumidor = np.clip(indice_confianca_consumidor, a_min=60, a_max=110)
        
        variacao_producao_industrial = np.random.normal(loc=0.3, scale=2.5, size=84)
        
        Processador.dados_brutos = pd.DataFrame({
            'data': datas,
            'ipca_mensal_percentuais': ipca_mensal_percentuais,
            'ipca_acumulado_doze_meses': ipca_acumulado_doze_meses,
            'taxa_desemprego_percentual': taxa_desemprego_percentual,
            'indice_confianca_consumidor': indice_confianca_consumidor,
            'variacao_producao_industrial': variacao_producao_industrial
        })
        
        Processador.metadados['data_carregamento'] = datetime.now()
        Processador.metadados['total_observacoes'] = len(Processador.dados_brutos)
        Processador.metadados['periodo_inicio'] = str(datas[0].date())
        Processador.metadados['periodo_fim'] = str(datas[-1].date())
        
        return Processador.dados_brutos.copy()
    
    def validar_qualidade() -> Dict:
        if Processador.dados_brutos is None:
            raise ValueError("Carregue dados primeiro com carregar_dados_amostra()")
        
        total_linhas = len(Processador.dados_brutos)
        duplicatas = Processador.dados_brutos.duplicated().sum()
        ausentes = Processador.dados_brutos.isnull().sum().sum()
        
        colunas_numericas = Processador.dados_brutos.select_dtypes(include=[np.number]).columns
        
        outliers_resultado = {}
        for coluna in colunas_numericas:
            q1 = Processador.dados_brutos[coluna].quantile(0.25)
            q3 = Processador.dados_brutos[coluna].quantile(0.75)
            amplitude = q3 - q1
            limite_inferior = q1 - 1.5 * amplitude
            limite_superior = q3 + 1.5 * amplitude
            
            outliers = (
                (Processador.dados_brutos[coluna] < limite_inferior) |
                (Processador.dados_brutos[coluna] > limite_superior)
            ).sum()
            
            outliers_resultado[coluna] = {
                'quantidade': int(outliers),
                'percentual': round((outliers / total_linhas) * 100, 2)
            }
        
        Processador.validacao = {
            'total_registros': total_linhas,
            'duplicatas': int(duplicatas),
            'valores_ausentes': int(ausentes),
            'status': 'Válido' if (duplicatas == 0 and ausentes == 0) else 'Revisar',
            'outliers_por_coluna': outliers_resultado
        }
        
        return Processador.validacao
    
    def limpar_dados() -> pd.DataFrame:
        if Processador.dados_brutos is None:
            raise ValueError("Carregue dados primeiro")
        
        dados_limpos = Processador.dados_brutos.drop_duplicates(subset=['data'], keep='first')
        
        colunas_numericas = dados_limpos.select_dtypes(include=[np.number]).columns
        dados_limpos[colunas_numericas] = dados_limpos[colunas_numericas].fillna(method='ffill').fillna(method='bfill')
        
        Processador.dados_processados = dados_limpos
        return Processador.dados_processados.copy()
    
    def obter_metricas_inflacao() -> Dict[str, float]:
        if Processador.dados_brutos is None:
            raise ValueError("Carregue dados primeiro")
        
        ultimo = Processador.dados_brutos.iloc[-1]
        penultimo = Processador.dados_brutos.iloc[-2]
        
        ipca_atual = float(ultimo['ipca_mensal_percentuais'])
        ipca_anterior = float(penultimo['ipca_mensal_percentuais'])
        ipca_doze_meses = float(ultimo['ipca_acumulado_doze_meses'])
        
        media_historica = float(Processador.dados_brutos['ipca_mensal_percentuais'].mean())
        desvio_padrao = float(Processador.dados_brutos['ipca_mensal_percentuais'].std())
        
        variacao_pct = (
            ((ipca_atual - ipca_anterior) / ipca_anterior * 100)
            if ipca_anterior > 0 else 0
        )
        
        return {
            'ipca_mensal_atual': round(ipca_atual, 3),
            'ipca_doze_meses': round(ipca_doze_meses, 2),
            'variacao_mensal_percentual': round(variacao_pct, 2),
            'media_historica': round(media_historica, 3),
            'desvio_padrao': round(desvio_padrao, 3),
            'minimo': round(Processador.dados_brutos['ipca_mensal_percentuais'].min(), 3),
            'maximo': round(Processador.dados_brutos['ipca_mensal_percentuais'].max(), 3)
        }
    
    def prever_arima_simples(periodos: int = 6) -> pd.DataFrame:
        if Processador.dados_brutos is None:
            raise ValueError("Carregue dados primeiro")
        
        media_recente = Processador.dados_brutos['ipca_mensal_percentuais'].tail(12).mean()
        desvio_recente = Processador.dados_brutos['ipca_mensal_percentuais'].tail(12).std()
        
        valores_previstos = []
        for _ in range(periodos):
            valor = np.random.normal(media_recente, desvio_recente)
            valores_previstos.append(max(0, valor))
        
        previsoes = pd.DataFrame({
            'horizonte_meses': range(1, periodos + 1),
            'ipca_previsto': valores_previstos,
            'intervalo_inferior': [
                max(0, v - 1.96 * desvio_recente) for v in valores_previstos
            ],
            'intervalo_superior': [
                v + 1.96 * desvio_recente for v in valores_previstos
            ]
        })
        
        return previsoes
    
    def exportar_para_csv(nome_arquivo: str = 'indicadores.csv') -> str:
        if Processador.dados_brutos is None:
            raise ValueError("Nenhum dado para exportar")
        
        Processador.dados_brutos.to_csv(nome_arquivo, index=False, encoding='utf-8')
        return nome_arquivo
    
    def exportar_para_excel(nome_arquivo: str = 'indicadores.xlsx') -> str:
        if Processador.dados_brutos is None:
            raise ValueError("Nenhum dado para exportar")
        
        with pd.ExcelWriter(nome_arquivo, engine='openpyxl') as escritor:
            Processador.dados_brutos.to_excel(escritor, sheet_name='Dados Brutos', index=False)
            
            if Processador.dados_processados is not None:
                Processador.dados_processados.to_excel(
                    escritor, sheet_name='Dados Processados', index=False
                )
        
        return nome_arquivo


if __name__ == "__main__":
    proc = Processador()
    dados = proc.carregar_dados_amostra()
    proc.validar_qualidade()
    proc.limpar_dados()
    metricas = proc.obter_metricas_inflacao()
    previsoes = proc.prever_arima_simples(periodos=6)
    proc.exportar_para_excel('indicadores.xlsx')
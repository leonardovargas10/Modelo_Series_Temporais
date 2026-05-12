﻿---

## Modelo de Forecasting de Temperatura — Daily Climate Delhi

<p align="center">
  <img src="./img01.jpg" width="50%">
</p>

> **Autor:** Leonardo Aderaldo Vargas  
> **Fonte dos Dados:** [Kaggle — Daily Climate Time Series Data](https://www.kaggle.com/datasets/sumanthvrao/daily-climate-time-series-data?select=DailyDelhiClimateTrain.csv)  
> **Status:**
<p align="center">
<img src="http://img.shields.io/static/v1?label=STATUS&message=DESENVOLVIMENTO&color=GREEN&style=for-the-badge"/>
</p>

---

## Sumário

1. [Contexto do Projeto](#1-contexto-do-projeto)
2. [Objetivos](#2-objetivos)
3. [Fundamentação Teórica](#3-fundamentação-teórica)
4. [Fonte de Dados](#4-fonte-de-dados)
5. [Arquitetura da Solução](#5-arquitetura-da-solução)
6. [Definição da Target](#6-definição-da-target)
7. [Análise Exploratória](#7-análise-exploratória)
8. [Amostragem](#8-amostragem)
9. [Pré-Processamento e Feature Engineering](#9-pré-processamento-e-feature-engineering)
10. [Modelagem](#10-modelagem)
11. [Resultados Consolidados](#11-resultados-consolidados)
12. [Artefatos Gerados](#12-artefatos-gerados)
13. [Referências](#13-referências)

---

## 1. Contexto do Projeto

Este projeto tem como objetivo estudar diferentes abordagens de **Séries Temporais** para previsão da temperatura média diária em Delhi, Índia.

A previsão de temperatura é um problema clássico de forecasting e permite comparar modelos estatísticos, modelos de Machine Learning e redes neurais recorrentes em uma série com tendência, sazonalidade, autocorrelação e variáveis meteorológicas auxiliares.

---

## 2. Objetivos

- Prever a variável `meantemp`, correspondente à temperatura média diária;
- Analisar tendência, sazonalidade, ruído e estacionaridade da série;
- Comparar modelos estatísticos, modelos de árvore e Deep Learning;
- Avaliar o impacto de lags e médias móveis como variáveis preditivas;
- Comparar desempenho via MAE e RMSE;
- Entender as limitações de ARIMA/SARIMA frente a modelos supervisionados com features temporais.

---

## 3. Fundamentação Teórica

- [x] Fundamentos de GIT
- [x] SQL
- [x] Python
- [x] Análise de Dados
- [x] Técnicas de Machine Learning
- [x] Técnicas de Séries Temporais
- [x] Modelagem Matemática e Estatística
- [x] ARIMA e SARIMA
- [x] Feature Engineering Temporal
- [x] Redes Neurais Recorrentes — LSTM

---

## 4. Fonte de Dados

A base utilizada é o dataset **Daily Climate Time Series Data**, disponível no Kaggle.

| Campo | Descrição |
|---|---|
| date | Data da medição |
| meantemp | Temperatura média diária |
| humidity | Umidade |
| wind_speed | Velocidade do vento |
| meanpressure | Pressão atmosférica |

A base `DailyDelhiClimateTrain.csv` possui registros diários entre **2013-01-01 e 2017-01-01**.

---

## 5. Arquitetura da Solução

    Dados Brutos
    DailyDelhiClimateTrain.csv
            |
            v
    Análise Exploratória
    Distribuições, outliers, tendência, sazonalidade e estacionaridade
            |
            v
    Pré-Processamento
    Tratamento de outliers em pressão atmosférica
            |
            v
    Feature Engineering Temporal
    Lags, médias móveis, diferenciação e transformação log
            |
            v
    Separação Temporal
    Treino / Validação / Teste
            |
            v
    Modelos Estatísticos
    ARIMA e SARIMA
            |
            v
    Modelos de Machine Learning
    Random Forest e LightGBM
            |
            v
    Modelo de Deep Learning
    LSTM com janela de 30 dias
            |
            v
    Avaliação
    MAE e RMSE por amostra

---

## 6. Definição da Target

A variável-resposta do projeto é:

```text
meantemp
```

Ela representa a **temperatura média diária**, calculada a partir de medições em diferentes intervalos do dia.

O objetivo é prever a temperatura média em datas futuras utilizando:

- comportamento passado da própria temperatura;
- umidade;
- velocidade do vento;
- pressão atmosférica;
- lags temporais;
- médias móveis.

---

## 7. Análise Exploratória

### 7.1 Distribuição das Variáveis

Principais observações:

- as distribuições de temperatura, umidade e velocidade do vento não apresentam outliers extremos relevantes;
- a variável `meanpressure` possui valores claramente inconsistentes, como pressão negativa e valor máximo próximo de 7679;
- não foram identificados dados ausentes;
- em caso de nulos, a estratégia sugerida seria imputação por média móvel de períodos anteriores.

### 7.2 Comportamento Temporal

A análise temporal indicou que:

- `meantemp` apresenta ciclos, tendência e sazonalidade;
- `humidity` também possui comportamento cíclico;
- `wind_speed` aparenta ser mais estacionária visualmente;
- `meanpressure`, após remoção de outliers extremos, apresenta comportamento inversamente relacionado à temperatura.

### 7.3 Decomposição da Série

Foi aplicada decomposição aditiva:

```text
Yt = Tendência + Sazonalidade + Resíduo
```

A série de temperatura apresenta componentes bem definidas de tendência e sazonalidade.

### 7.4 Estacionaridade

Foram utilizados os testes:

| Teste | Hipótese Nula |
|---|---|
| ADF — Dickey-Fuller Aumentado | A série não é estacionária |
| KPSS | A série é estacionária |

Resultados principais:

- a série original de `meantemp` não foi considerada estacionária pelo ADF;
- após diferenciação, a série se torna estatisticamente estacionária;
- a diferenciação de segunda ordem foi analisada como alternativa para ARIMA;
- mesmo com diferenciação, a série ainda apresenta variância elevada e sinais de sazonalidade.

---

## 8. Amostragem

A separação foi temporal:

| Conjunto | Período Original | Uso |
|---|---|---|
| Treino | 2015-01-01 a 2016-08-31 | Treinamento dos modelos |
| Validação | 2016-09-01 a 2016-10-31 | Avaliação intermediária |
| Teste | 2016-11-01 a 2016-12-31 | Avaliação final |

Após criação de lags e médias móveis, os registros iniciais sem histórico suficiente foram removidos.

| Conjunto | Registros após Feature Engineering |
|---|---:|
| Treino | 429 |
| Validação | 61 |
| Teste | 61 |

---

## 9. Pré-Processamento e Feature Engineering

### 9.1 Tratamento de Outliers

A variável `meanpressure` apresentou valores fisicamente inconsistentes. Foram mantidos valores entre 994 e 1020, substituindo valores extremos pela mediana da pressão atmosférica no treino.

### 9.2 Transformações da Target

Foram criadas variações da série:

| Feature | Descrição |
|---|---|
| meantemp_log | Log da temperatura média |
| meantemp_diff2 | Diferenciação de segunda ordem |
| meantemp_log_diff2 | Log com diferenciação de segunda ordem |
| meantemp_seasonal_diff | Diferença sazonal de 7 dias |

### 9.3 Lags Temporais

Foram criados lags para temperatura, umidade, vento e pressão:

| Janela | Descrição |
|---|---|
| d1 | Lag de 1 dia |
| d2 | Lag de 2 dias |
| d3 | Lag de 3 dias |
| m1 | Lag de 30 dias |
| m3 | Lag de 90 dias |
| m6 | Lag de 180 dias |

### 9.4 Médias Móveis

Foram criadas médias móveis com defasagem para evitar data leakage:

| Janela | Descrição |
|---|---|
| d3 | Média móvel de 3 dias |
| m1 | Média móvel de 30 dias |
| m3 | Média móvel de 90 dias |
| m6 | Média móvel de 180 dias |

---

## 10. Modelagem

Foram avaliadas três famílias de modelos.

### 10.1 Modelos Estatísticos

| Modelo | Configuração |
|---|---|
| ARIMA | ARIMA(1, 2, 1) |
| SARIMA | SARIMA(1, 2, 1)(1, 1, 1, 7) |

O ARIMA foi escolhido com base na análise de ACF/PACF e na estacionarização via diferenciação.

O SARIMA foi testado para capturar sazonalidade semanal, usando período sazonal de 7 dias.

### 10.2 Modelos de Machine Learning

| Modelo | Configuração |
|---|---|
| Random Forest Regressor | n_estimators=50, max_depth=4 |
| LightGBM Regressor | n_estimators=50, max_depth=4, learning_rate=0,1 |

A seleção de variáveis foi feita com **Random Forest Feature Importance** e remoção de features correlacionadas.

### 10.3 Modelo de Deep Learning

Foi treinada uma LSTM com:

| Parâmetro | Valor |
|---|---:|
| Lags | 30 dias |
| Épocas | 50 |
| Batch Size | 32 |
| Neurônios LSTM | 64 |
| Dropout | 0,2 |
| Otimizador | Adam |
| Loss | MSE |
| Early Stopping | patience=5 |

---

## 11. Resultados Consolidados

### 11.1 Métricas por Modelo

| Modelo | Etapa | MAE | RMSE | AIC |
|---|---|---:|---:|---:|
| ARIMA(1,2,1) | Treino | 1,27 | 2,31 | 1627,05 |
| ARIMA(1,2,1) | Validação | 3,27 | 3,62 | 1627,05 |
| ARIMA(1,2,1) | Teste | 6,40 | 7,04 | 1627,05 |
| SARIMA(1,2,1)(1,1,1,7) | Treino | 1,47 | 2,77 | 1580,45 |
| SARIMA(1,2,1)(1,1,1,7) | Validação | 3,87 | 4,20 | 1580,45 |
| SARIMA(1,2,1)(1,1,1,7) | Teste | 3,58 | 4,29 | 1580,45 |
| Random Forest | Treino | 0,97 | 1,27 | — |
| Random Forest | Validação | 0,85 | 1,02 | — |
| Random Forest | Teste | 1,18 | 1,54 | — |
| LightGBM | Treino | 0,76 | 0,99 | — |
| LightGBM | Validação | 0,93 | 1,08 | — |
| LightGBM | Teste | 1,21 | 1,51 | — |
| LSTM | Treino | 1,93 | 2,43 | — |
| LSTM | Validação | 0,91 | 1,21 | — |
| LSTM | Teste | 2,21 | 2,54 | — |

### 11.2 Interpretação

Os modelos estatísticos foram úteis para compreender estrutura, estacionaridade e sazonalidade da série, mas apresentaram maior erro no período de teste.

Os modelos de Machine Learning, especialmente **Random Forest** e **LightGBM**, apresentaram desempenho mais consistente, aproveitando bem as features de lags, médias móveis e covariáveis meteorológicas.

A LSTM apresentou resultado satisfatório, mas com pior desempenho no teste em relação aos modelos de árvore, possivelmente pelo tamanho limitado da série após janelamento.

### 11.3 Modelo com Melhor Desempenho

Considerando o conjunto de teste:

| Critério | Melhor Modelo |
|---|---|
| Menor MAE | Random Forest |
| Menor RMSE | LightGBM |

O LightGBM apresentou o menor RMSE no teste, enquanto a Random Forest apresentou o menor MAE.

---

## 12. Artefatos Gerados

| Artefato | Localização | Descrição |
|---|---|---|
| DailyDelhiClimateTrain.csv | data/ | Base principal usada no estudo |
| DailyDelhiClimateTest.csv | data/ | Base externa disponibilizada pelo Kaggle |
| Modelo_Forecasting.ipynb | raiz | Notebook completo de EDA, feature engineering e modelagem |
| img01.jpg | raiz | Imagem utilizada no README |
| output.png | raiz | Imagem gerada durante as análises |
| Captura de tela 2023-05-31 215605.png | raiz | Imagem auxiliar do projeto |

---

## 13. Referências

[Kaggle — Daily Climate Time Series Data](https://www.kaggle.com/datasets/sumanthvrao/daily-climate-time-series-data?select=DailyDelhiClimateTrain.csv)

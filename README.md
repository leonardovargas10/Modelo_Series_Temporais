---

## Modelo de Forecasting de Temperatura — Daily Climate Delhi

<p align="center">
  <img src="./img01.jpg" width="50%">
</p>

> **Autor:** Leonardo Aderaldo Vargas  
> **Fonte dos Dados:** [Kaggle — Daily Climate Time Series Data](https://www.kaggle.com/datasets/sumanthvrao/daily-climate-time-series-data?select=DailyDelhiClimateTrain.csv)  

> **Status:**

<p align="center">
<img src="http://img.shields.io/static/v1?label=STATUS&message=CONCLUIDO&color=GREEN&style=for-the-badge"/>
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

Este projeto estuda diferentes abordagens de **Séries Temporais** para previsão da temperatura média diária em Delhi, Índia.

A tarefa foi definida como uma previsão de curto prazo: ao final do dia **D**, deseja-se estimar a temperatura média do dia **D+1**. Essa definição permite comparar modelos estatísticos, modelos de Machine Learning e redes neurais recorrentes sob o mesmo horizonte e utilizando apenas informações disponíveis no instante da previsão.

A série apresenta ciclo anual, forte persistência temporal, variabilidade ao longo das estações e variáveis meteorológicas auxiliares. O principal desafio é verificar se modelos complexos conseguem superar, de forma consistente, uma regra simples: assumir que a temperatura de amanhã será igual à temperatura observada hoje.

---

## 2. Objetivos

- Prever `meantemp`, correspondente à temperatura média diária de D+1;
- Analisar distribuição, tendência, sazonalidade, resíduos, estacionaridade e autocorrelação;
- Comparar ARIMA, SARIMA, Random Forest, LightGBM e LSTM;
- Utilizar persistência D-1 e sazonal ingênuo D-7 como baselines;
- Garantir que todos os modelos sejam avaliados nas mesmas datas e no mesmo horizonte;
- Construir features temporais estritamente causais;
- Avaliar generalização por validação temporal walk-forward;
- Comparar desempenho via MAE, RMSE, MASE e sMAPE;
- Interpretar as features por permutação fora do treino e por ablação;
- Quantificar a incerteza do ganho sobre o baseline por bootstrap em blocos.

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
- [x] Validação Walk-Forward
- [x] Métricas MASE e sMAPE
- [x] Importância por Permutação e Ablação
- [x] Bootstrap em Blocos
- [x] Redes Neurais Recorrentes — LSTM

---

## 4. Fonte de Dados

A base utilizada é o dataset **Daily Climate Time Series Data**, disponível no Kaggle.

| Campo | Descrição |
|---|---|
| date | Data da medição |
| meantemp | Temperatura média diária |
| humidity | Umidade média diária |
| wind_speed | Velocidade média diária do vento |
| meanpressure | Pressão atmosférica média diária |

Os arquivos públicos possuem a data `2017-01-01` tanto no desenvolvimento quanto no teste, com valores diferentes. Para preservar o holdout oficial, foi mantida a observação pertencente ao teste e o desenvolvimento foi encerrado em `2016-12-31`.

| Amostra | Período | Observações | Uso |
|---|---|---:|---|
| Desenvolvimento | 2013-01-01 a 2016-12-31 | 1.461 | EDA, treinamento e validação temporal |
| Teste oficial | 2017-01-01 a 2017-04-24 | 114 | Avaliação final fora da amostra |

---

## 5. Arquitetura da Solução

    Dados Brutos
    DailyDelhiClimateTrain.csv + DailyDelhiClimateTest.csv
            |
            v
    Separação Oficial
    Desenvolvimento até 2016-12-31 / Teste em 2017
            |
            v
    Análise Exploratória no Desenvolvimento
    Distribuições, outliers, tendência, sazonalidade e estacionaridade
            |
            v
    Definição do Horizonte
    Previsão D+1 com informações disponíveis até D
            |
            v
    Feature Engineering Causal
    Lags, estatísticas móveis, calendário cíclico e tendência
            |
            v
    Validação Walk-Forward
    Quatro folds trimestrais de 2016 com janela expansiva
            |
            v
    Baselines
    Persistência D-1 e Sazonal Ingênuo D-7
            |
            v
    Modelos Estatísticos
    ARIMA e SARIMA com atualização diária do estado
            |
            v
    Modelos de Machine Learning
    Random Forest e LightGBM regularizados
            |
            v
    Modelo de Deep Learning
    LSTM com janela de 30 dias e validação interna
            |
            v
    Avaliação Final
    MAE, RMSE, MASE, sMAPE, resíduos e bootstrap em blocos

---

## 6. Definição da Target

A variável-resposta do projeto é:

```text
meantemp
```

Ela representa a **temperatura média diária**, calculada a partir de medições em diferentes intervalos do dia.

O objetivo é prever:

```text
temperatura média do dia D+1
```

No instante da previsão, estão disponíveis:

- comportamento passado da própria temperatura;
- lags e estatísticas móveis calculados até D;
- dia do ano de D+1;
- tendência temporal.

Umidade, vento e pressão são utilizados na análise exploratória, mas não entram no benchmark principal. Seus valores contemporâneos de D+1 ainda não seriam conhecidos sem uma previsão meteorológica externa dessas variáveis.

---

## 7. Análise Exploratória

### 7.1 Distribuição das Variáveis

Foram avaliados histogramas, boxplots e QQ Plots.

Principais observações:

- a temperatura apresenta distribuição multimodal em razão das estações do ano;
- umidade possui distribuição relativamente próxima da simetria, com desvios nas caudas;
- velocidade do vento apresenta forte assimetria à direita;
- `meanpressure` contém observações incompatíveis com a faixa principal, sugerindo erros de medição;
- não foram encontrados dados ausentes no desenvolvimento;
- valores extremos meteorológicos não foram removidos automaticamente, pois podem representar eventos reais.

### 7.2 Comportamento Temporal

A análise diária e as estatísticas móveis de 30 dias indicaram que:

- `meantemp` possui ciclo anual bem definido;
- `humidity` também apresenta comportamento cíclico;
- `wind_speed` possui dinâmica mais irregular;
- `meanpressure`, desconsiderando extremos apenas na visualização, apresenta movimento aproximadamente inverso à temperatura em parte do ano.

### 7.3 Decomposição da Série

Foi aplicada decomposição aditiva anual:

```text
Yt = Tendência + Sazonalidade + Resíduo
```

A decomposição com período de 365 dias evidencia uma componente sazonal anual forte. Ela foi utilizada de forma descritiva, não como regra automática para escolher a ordem do SARIMA.

### 7.4 Estacionaridade

Foram utilizados os testes:

| Teste | Hipótese Nula | Evidência favorável à estacionaridade |
|---|---|---|
| ADF — Dickey-Fuller Aumentado | A série possui raiz unitária | p-valor < 0,05 |
| KPSS | A série é estacionária | p-valor > 0,05 |

Foram comparadas a série em nível, a primeira diferença e a diferença anual. A primeira diferença reduz a persistência e produz comportamento mais próximo da estacionaridade.

Não foi aplicada uma segunda diferença automaticamente, pois diferenciação excessiva pode amplificar ruído, criar autocorrelação artificial e remover estrutura útil para previsão.

### 7.5 Persistência Temporal

A temperatura apresenta correlação elevada com seus primeiros lags, especialmente com D-1. Esse resultado é esperado fisicamente: a temperatura normalmente varia de forma gradual entre dias consecutivos.

Correlação elevada com lag 1 não representa, por si só, vazamento. Sua contribuição foi investigada posteriormente por importância por permutação e ablação.

---

## 8. Amostragem

A separação foi exclusivamente temporal, utilizando quatro folds trimestrais de 2016 com janela expansiva.

| Fold | Treinamento | Validação |
|---|---|---|
| 2016-Q1 | Todo o histórico anterior | 2016-01-01 a 2016-03-31 |
| 2016-Q2 | Todo o histórico anterior | 2016-04-01 a 2016-06-30 |
| 2016-Q3 | Todo o histórico anterior | 2016-07-01 a 2016-09-30 |
| 2016-Q4 | Todo o histórico anterior | 2016-10-01 a 2016-12-31 |

Em cada fold:

1. o modelo é ajustado somente com o passado;
2. é realizada a previsão do próximo dia;
3. o valor realizado é incorporado ao histórico;
4. o processo avança para o dia seguinte.

O teste oficial de 2017 permanece separado até que horizonte, features, ordens, hiperparâmetros e métricas estejam definidos.

---

## 9. Pré-Processamento e Feature Engineering

### 9.1 Tratamento de Outliers

Os valores extremos de pressão foram mantidos na análise descritiva e destacados nas visualizações. Como o benchmark D+1 utiliza apenas histórico da temperatura e calendário, esses valores não afetam diretamente os modelos.

Isso evita impor limites meteorológicos arbitrários ou substituir possíveis eventos reais sem uma fonte externa para confirmação.

### 9.2 Transformações da Target

As transformações foram usadas para diagnóstico de estacionaridade e seleção dos modelos estatísticos.

| Transformação | Objetivo |
|---|---|
| Série em nível | Preservar a escala original em °C |
| Primeira diferença | Investigar mudança diária e reduzir persistência |
| Diferença anual | Investigar repetição do ciclo de 365 dias |

A target dos modelos supervisionados permanece na escala original, facilitando interpretação e evitando reconstrução após diferenciação.

### 9.3 Lags Temporais

Foram criados lags da temperatura:

| Lag | Interpretação |
|---|---|
| 1 | Temperatura observada no dia anterior |
| 2 | Temperatura de dois dias atrás |
| 3 | Temperatura de três dias atrás |
| 7 | Mesmo dia da semana anterior |
| 14 | Duas semanas atrás |
| 30 | Aproximadamente um mês atrás |

### 9.4 Estatísticas Móveis e Calendário

| Grupo | Features |
|---|---|
| Médias móveis | 3, 7, 14 e 30 dias |
| Desvios móveis | 3, 7, 14 e 30 dias |
| Calendário | seno e cosseno do dia do ano |
| Longo prazo | tendência acumulada em dias |

Todas as estatísticas móveis são calculadas após `shift(1)`. Portanto, nenhuma feature da linha D+1 contém a própria temperatura de D+1.

---

## 10. Modelagem

Foram avaliadas três famílias de modelos, além de dois baselines.

### 10.1 Baselines

| Modelo | Regra |
|---|---|
| Persistência D-1 | Previsão de amanhã = temperatura observada hoje |
| Sazonal Ingênuo D-7 | Previsão de amanhã = temperatura observada sete dias atrás |

Os modelos complexos precisam superar essas regras para demonstrarem ganho preditivo real.

### 10.2 Modelos Estatísticos

As ordens foram selecionadas por AIC usando somente dados anteriores aos folds de validação.

| Modelo | Configuração selecionada |
|---|---|
| ARIMA | ARIMA(1, 1, 1) |
| SARIMA | SARIMA(1, 1, 1)(0, 0, 1, 7) |

O período sazonal 7 representa sazonalidade semanal. Um SARIMA com período 365 seria pouco parcimonioso para aproximadamente quatro ciclos anuais disponíveis.

Dentro de cada fold, os parâmetros permanecem fixos, mas o estado do modelo é atualizado diariamente com o valor realizado. Dessa forma, ARIMA e SARIMA também resolvem uma previsão D+1 walk-forward.

### 10.3 Modelos de Machine Learning

| Modelo | Configuração principal |
|---|---|
| Random Forest Regressor | 400 árvores, profundidade máxima 8, mínimo 5 observações por folha |
| LightGBM Regressor | 300 árvores, learning rate 0,03, 15 folhas, profundidade máxima 5 |

Os dois modelos foram regularizados para controlar variância.

A interpretação das features foi realizada por:

- importância por permutação no quarto fold;
- ablação com todas as features;
- ablação sem lag 1;
- modelo utilizando somente lag 1.

### 10.4 Modelo de Deep Learning

Foi treinada uma LSTM com:

| Parâmetro | Valor |
|---|---:|
| Janela | 30 dias |
| Épocas máximas | 60 |
| Batch Size | 32 |
| Unidades LSTM | 32 |
| Dropout | 0,20 |
| Otimizador | Adam |
| Loss | MSE |
| Early Stopping | patience=6 |

Os 15% finais de cada treino formam uma validação interna. Após selecionar o número de épocas, a rede é reinicializada e treinada novamente em todo o treino externo.

As janelas atravessam corretamente as fronteiras: o histórico anterior ao fold permanece disponível para prever o primeiro dia da validação ou do teste.

---

## 11. Resultados Consolidados

### 11.1 Validação Temporal

Média e desvio-padrão nos quatro folds trimestrais de 2016:

| Modelo | MAE Médio | Desvio MAE | RMSE Médio | MASE Médio | sMAPE Médio |
|---|---:|---:|---:|---:|---:|
| Persistência D-1 | **1,227** | 0,183 | 1,625 | **0,996** | **4,925%** |
| Random Forest | 1,259 | 0,311 | **1,592** | 1,024 | 4,965% |
| LightGBM | 1,267 | 0,295 | 1,619 | 1,030 | 5,003% |
| SARIMA | 1,268 | **0,175** | 1,623 | 1,030 | 5,114% |
| ARIMA | 1,366 | 0,174 | 2,110 | 1,110 | 5,505% |
| LSTM | 1,599 | 0,420 | 2,012 | 1,300 | 6,461% |
| Sazonal Ingênuo D-7 | 2,275 | 0,344 | 2,824 | 1,849 | 9,162% |

A persistência D-1 apresentou o menor MAE médio. Random Forest, LightGBM e SARIMA ficaram próximos, mas não produziram uma melhora consistente sobre o baseline.

### 11.2 Teste Oficial

Resultados no período de 2017, aberto somente após a definição completa do experimento:

| Modelo | MAE | RMSE | MASE | sMAPE | Skill vs. Persistência |
|---|---:|---:|---:|---:|---:|
| Random Forest | **1,300** | **1,647** | **1,061** | **6,495%** | 1,011% |
| Persistência D-1 | 1,314 | 1,684 | 1,072 | 6,615% | 0,000% |
| ARIMA | 1,335 | 1,681 | 1,089 | 6,624% | -1,589% |
| LightGBM | 1,355 | 1,686 | 1,106 | 6,786% | -3,184% |
| SARIMA | 1,512 | 1,893 | 1,234 | 7,251% | -15,069% |
| LSTM | 1,743 | 2,118 | 1,422 | 8,830% | -32,655% |
| Sazonal Ingênuo D-7 | 2,978 | 3,663 | 2,430 | 14,681% | -126,664% |

### 11.3 Interpretação

A Random Forest obteve o menor MAE no teste, superando a persistência por aproximadamente `0,013 °C`, equivalente a cerca de 1%.

Entretanto:

- a Random Forest não venceu a validação temporal;
- a diferença absoluta é muito pequena;
- o bootstrap em blocos estimou intervalo de 95% aproximadamente entre `-0,096 °C` e `0,112 °C` para seu ganho de MAE;
- como o intervalo inclui zero, não existe evidência robusta de superioridade.

Escolher a Random Forest apenas por liderar o teste transformaria o holdout em uma segunda etapa de seleção.

### 11.4 Importância do Lag 1

O lag 1 é importante porque a temperatura possui continuidade física. Porém, não representa literalmente 100% da informação.

Na ablação do quarto fold:

| Modelo | Features | MAE |
|---|---|---:|
| Random Forest | Todas | **0,922** |
| Random Forest | Sem lag 1 | 0,965 |
| Random Forest | Somente lag 1 | 1,074 |
| LightGBM | Todas | **0,964** |
| LightGBM | Sem lag 1 | 0,974 |
| LightGBM | Somente lag 1 | 1,059 |

Ao retirar lag 1, lags próximos e médias móveis recuperam parte da informação. Isso mostra por que uma importância concentrada em uma única feature não deve ser interpretada como exclusividade causal.

### 11.5 Modelo Recomendado

| Critério | Resultado |
|---|---|
| Menor MAE médio na validação | Persistência D-1 |
| Menor MAE nominal no teste | Random Forest |
| Ganho robusto sobre persistência | Nenhum modelo |
| Benchmark operacional recomendado | **Persistência D-1** |

Para previsão D+1 sem previsões meteorológicas externas, a persistência é a decisão mais simples e defensável. Random Forest, LightGBM e SARIMA permanecem como candidatos de pesquisa, mas não demonstraram ganho estável suficiente para justificar maior complexidade.

Essa conclusão é específica para D+1. Previsões D+7 ou D+30 exigiriam um novo experimento, com features deslocadas pelo horizonte e avaliação direta ou recursiva.

---

## 12. Artefatos Gerados

| Artefato | Localização | Descrição |
|---|---|---|
| DailyDelhiClimateTrain.csv | data/ | Base de desenvolvimento disponibilizada pelo Kaggle |
| DailyDelhiClimateTest.csv | data/ | Holdout oficial de 2017 |
| Modelo_Forecasting.ipynb | raiz | Notebook autocontido com EDA, metodologia, modelos e resultados |
| img01.jpg | raiz | Imagem utilizada no README |
| output.png | raiz | Imagem auxiliar gerada durante as análises |
| Captura de tela 2023-05-31 215605.png | raiz | Imagem histórica do projeto |

---

## 13. Referências

- [Kaggle — Daily Climate Time Series Data](https://www.kaggle.com/datasets/sumanthvrao/daily-climate-time-series-data?select=DailyDelhiClimateTrain.csv)
- [Statsmodels — ARIMA](https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima.model.ARIMA.html)
- [Statsmodels — SARIMAX](https://www.statsmodels.org/stable/generated/statsmodels.tsa.statespace.sarimax.SARIMAX.html)
- [Scikit-learn — Permutation Importance](https://scikit-learn.org/stable/modules/permutation_importance.html)

# STTS Imputation Framework – Experimentos do TCC

Este repositório contém o código-fonte, dados e componentes do framework desenvolvido para o Trabalho de Conclusão de Curso de **Bruno Duarte Maia**, intitulado **"Framework para Imputação de Dados Faltantes em Séries Espaço-Temporais de Sensores Físicos"**. O trabalho propõe uma ferramenta integrada para avaliar métodos de imputação aplicados a séries espaço-temporais, com foco em sensores ambientais e modelos numéricos.

## Introdução

O projeto apresenta um framework capaz de gerar cenários de falhas em séries espaço-temporais e avaliar a capacidade de diferentes métodos de reconstruir valores ausentes. A ferramenta combina backend em Python (FastAPI) e frontend em React, permitindo explorar visualmente mapas, séries temporais e métricas de erro. O objetivo central é oferecer uma plataforma reprodutível para comparar métodos estatísticos e de deep learning em cenários reais e controlados de sensores físicos.

## Datasets

Os experimentos utilizam dois conjuntos de dados:

- **ROMS – Cabo Frio (SSH)**: séries temporais da elevação da superfície do mar obtidas de simulações numéricas, formando um conjunto completo e adequado para gerar falhas artificiais e testar métodos de imputação em condições controladas.

- **Beijing AQI-36 (PM2.5)**: medições reais de PM2.5 provenientes de 36 estações urbanas, oferecendo um cenário com variabilidade espacial e ocorrência natural de falhas, ideal para avaliar a robustez dos métodos em ambiente observacional.

## Métodos Testados

Os métodos avaliados incluem técnicas estatísticas e de aprendizado profundo aplicadas à tarefa de imputação:

- SPIN  
- PriSTI  
- KNN  
- ARIMA  

Cada método é executado sobre séries mascaradas segundo diferentes padrões de falha, permitindo comparar sua precisão e robustez.

## Metodologia

A metodologia consiste em selecionar um dataset, gerar máscaras artificiais de falha (aleatórias, em blocos ou espacialmente concentradas), aplicar os métodos de imputação escolhidos e comparar os valores reconstruídos com o ground truth. As métricas utilizadas incluem MAE e MSE, complementadas por inspeção visual das séries imputadas por meio da interface interativa do framework.

## Resultados

Os resultados são analisados com base nas métricas de erro e na visualização das séries reconstruídas. O framework permite identificar diferenças claras entre os métodos, tanto em cenários controlados de simulação quanto em dados reais com irregularidades e falhas naturais.

## Execução

### Backend (FastAPI)

Dentro da pasta `backend/`:

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
Frontend (React + Vite)
Dentro da pasta frontend/:

bash
Copiar código
npm install
VITE_API_URL=http://localhost:8000 npm run dev

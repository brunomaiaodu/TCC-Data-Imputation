# STTS Imputation Framework – Experimentos do TCC

Este repositório contém o código-fonte, dados processados e componentes do framework desenvolvido para o Trabalho de Conclusão de Curso de **Bruno Maia de Oliveira Duarte**, intitulado **_Framework para Imputação de Dados Faltantes em Séries Espaço-Temporais de Sensores Físicos_**.

O projeto consiste em uma **ferramenta integrada para visualização, análise e comparação de experimentos de imputação previamente executados**, com foco em séries espaço-temporais provenientes de sensores físicos e modelos numéricos ambientais.

---

## Introdução

A perda de dados é um problema recorrente em sistemas de sensoriamento físico, seja por falhas de comunicação, manutenção de equipamentos ou limitações operacionais. No contexto de séries espaço-temporais, tais lacunas afetam diretamente tarefas de análise, previsão e tomada de decisão.

Este trabalho apresenta um framework que **centraliza e organiza os resultados de experimentos de imputação já realizados**, permitindo a **comparação sistemática entre métodos distintos** por meio de métricas quantitativas e visualizações interativas. Diferentemente de ferramentas focadas na execução dos algoritmos, o framework atua como uma **camada de análise e inspeção experimental**, apoiando estudos comparativos e reprodutíveis.

A aplicação combina um backend em Python (FastAPI) com um frontend em React, possibilitando a exploração conjunta de mapas, séries temporais e métricas de erro associadas aos experimentos.

---

## Datasets

Os experimentos disponibilizados no framework utilizam dois conjuntos de dados amplamente empregados na literatura:

- **ROMS – Cabo Frio (SSH)**  
  Séries temporais da elevação da superfície do mar (_Sea Surface Height_) obtidas a partir de simulações do *Regional Ocean Model System*. O dataset é completo e serve como *ground truth* para experimentos controlados, nos quais diferentes percentuais e padrões de falha foram previamente aplicados.

- **Beijing AQI-36 (PM2.5)**  
  Conjunto de dados reais de qualidade do ar contendo medições de PM2.5 em 36 estações urbanas de Pequim. O dataset apresenta variabilidade espaço-temporal significativa e falhas naturais, permitindo avaliar o comportamento dos métodos em cenários observacionais reais.

---

## Métodos Comparados

O framework disponibiliza resultados de experimentos previamente executados com os seguintes métodos de imputação:

- **SPIN (_Spatiotemporal Point Inference Network_)**
- **KNN (_K-Nearest Neighbours_)**

Outros métodos podem ser incorporados futuramente, desde que seus resultados sigam o formato de dados adotado pelo framework.

---

## Metodologia

A metodologia adotada no TCC consiste em:

1. Seleção do dataset e definição do *ground truth*;
2. Geração **prévia** de cenários de falha com diferentes percentuais e padrões (aleatórios, temporais ou espaciais);
3. Execução offline dos métodos de imputação;
4. Armazenamento dos resultados e métricas em arquivos padronizados;
5. Carregamento desses resultados no framework para:
   - comparação quantitativa (MAE, MSE, RMSE);
   - inspeção visual das séries temporais originais, mascaradas e imputadas;
   - análise comparativa entre métodos e níveis de falha.

O framework **não executa os métodos de imputação em tempo real**, atuando exclusivamente como uma **plataforma de análise experimental**.

---

## Resultados

A interface permite analisar os resultados sob diferentes perspectivas:

- comparação direta entre métodos para um mesmo percentual de falha;
- avaliação da degradação de desempenho conforme o aumento da taxa de dados ausentes;
- visualização das séries temporais imputadas em pontos específicos do domínio espacial;
- inspeção integrada de métricas globais e comportamento local das reconstruções.

## Como rodar localmente

### Backend (FastAPI)
```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn backend.app.main:app --reload --port 8000
```

### Frontend (Vite/React)
```bash
cd frontend
npm install
npm run dev -- --host --port 5173
# API padrão: http://localhost:8000 (ajuste com VITE_API_URL se necessário)
```

Abra o navegador em http://localhost:5173.

## Docker / Compose

Para subir tudo via Docker (Dockerfile único na raiz, com targets para backend/frontend):
```bash
docker compose up --build
```
- Backend: http://localhost:8000
- Frontend: http://localhost:4173 (apontando para o backend)

## Estrutura de dados
- `preprocessing/processed/roms_*` e `aqi36_*`: séries imputadas por percentual/método/nó.
- `backend/app/data/*_latlon_enriched.csv`: lat/lon das malhas para os mapas.

## Observações
- ROMS: seletor de meses restrito a março e junho; dados com passo de 2h.
- AQI-36: série horária e falha original fixa (~23%).

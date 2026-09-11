# Dashboard Vagem — Quimiometria de Bioadsorção (Palma × Vagem)

Painel interativo de análise de dados para um estudo de bioadsorção de corante catiônico (azul de metileno) usando fibras vegetais — **Vagem** e **Palma** — tratadas quimicamente de três formas: **In natura**, **Ácido (H₃PO₄)** e **Base (NaOH)**.

**🔗 Acesse online:** https://willpine1992.github.io/CHEMOMETRICS-PALMA-VAGEM/

> A versão publicada é estática (GitHub Pages). Todas as seções funcionam normalmente a partir de dados pré-gerados, **exceto o "Simulador Interativo"** da seção de Machine Learning, que precisa do backend Flask rodando localmente (`python ml/app.py`) para calcular predições sob demanda.

---

## 1. Contexto dos dados

Os dados brutos vêm de uma planilha de laboratório (Google Sheets) sincronizada ao vivo do Google Drive a cada execução do ETL — não é uma cópia estática. A planilha registra experimentos de adsorção "um fator por vez" (OFAT):

- **pH**, **massa de adsorvente**, **tempo de contato** e **concentração inicial / temperatura** variados separadamente, medindo a concentração de equilíbrio (Ce) por espectrofotometria (curva de calibração absorbância → concentração).
- Dois materiais adsorventes: fibra da **Palma** e fibra da **Vagem**, cada um em três tratamentos: **In natura** (sem tratamento), **Ácido** (imersão em H₃PO₄ 0,1 mol/L) e **Base** (imersão em NaOH 0,1 mol/L).
- A partir de Ce, calcula-se **Qₑ** (capacidade de adsorção, mg/g) e **%Remoção** — as duas variáveis-resposta usadas em todas as análises.

Esses dados alimentam um banco relacional SQLite (`db/biosorcao.db`), construído por um ETL (`etl/build_database.py`) que:

1. Baixa a planilha mais recente do Google Drive via API (service account).
2. Faz o parsing de cada aba (pH, Massa, Tempo, Concentração+Temperatura, Isotermas, Parâmetros) para as 4 combinações material×tratamento relevantes.
3. Valida a estrutura da planilha (nomes de aba, células-âncora, contagens de linha, faixas plausíveis) **antes** de substituir o banco de produção — se a planilha mudar de layout no Drive, o ETL falha alto em vez de gravar dados errados silenciosamente (ver `etl/validate.py`).
4. Recalcula Qₑ e %Remoção pela fórmula padrão (`Qe = (C0−Ce)·V/m`), pois as colunas de Qe da planilha original continham erros de fórmula em algumas abas.

Esse banco alimenta os quatro módulos de análise descritos abaixo, cada um com seus próprios scripts Python e artefatos (JSON/joblib/cube) que o dashboard consome.

---

## 2. Metodologia de cada análise

### 2.1 — Modelagem Preditiva: Machine Learning & XAI (`ml/`)

- **Dataset**: 49 corridas experimentais reais (Vagem, tratamento Ácido) — o mesmo recorte usado como referência nas demais análises de ML/RSM, para manter os resultados comparáveis entre si.
- **Pré-processamento**: padronização (`StandardScaler`) das 5 variáveis de entrada (pH, massa, tempo, concentração inicial, temperatura).
- **Divisão**: treino/teste 75/25, com `k-fold` (k=5) dentro do `GridSearchCV` para tuning de hiperparâmetros.
- **Modelos**: `RandomForestRegressor`, `XGBRegressor`, `SVR` (scikit-learn / xgboost), treinados para prever **Qₑ** e **%Remoção** separadamente. O melhor de cada alvo é escolhido pelo R² de teste (desempate por menor RMSE).
- **Explicabilidade**: valores SHAP (`shap.TreeExplainer` ou `KernelExplainer`, conforme o modelo) para gerar o gráfico de importância de variáveis (beeswarm).
- **Validação**: R², RMSE e MAE reportados separadamente para treino e teste — o dataset é pequeno, então o R² de teste é tratado como estimativa exploratória, não como métrica definitiva.
- Script: `ml/train_pipeline.py` · Backend de predição ao vivo: `ml/app.py`.

### 2.2 — Otimização Multivariada: DoE & RSM (`rsm/`)

- **Importante**: os dados são OFAT, **não** um Central Composite Design (CCD) ou Box-Behnken (BBD) desenhado a priori. O RSM aqui é aplicado retrospectivamente aos dados existentes — uma prática comum quando não há orçamento para um novo desenho experimental, mas com menos poder estatístico que um CCD/BBD de verdade.
- **Seleção de termos por VIF**: o modelo quadrático completo (21 termos: 5 lineares + 5 quadráticos + 10 interações) é matematicamente singular nesse dataset (correlação estrutural entre fatores OFAT, ex. massa e tempo com r=−0,89). Via *Variance Inflation Factor*, mantivemos só os termos com suporte real de dados (VIF < 12): 5 lineares + 5 quadráticos + 3 interações (pH×Massa, pH×Tempo, Concentração×Temperatura).
- **ANOVA (Tipo II)** e **gráfico de Pareto dos efeitos padronizados** (`statsmodels`) para testar a significância de cada termo.
- **Superfície de resposta 3D e mapa de contorno** (Plotly) para a única interação estatisticamente significativa (pH × Massa).
- **Otimização por desirability** (`scipy.optimize.differential_evolution`): busca o ponto que maximiza Qₑ previsto minimizando a massa de adsorvente, dentro da faixa observada — sinalizado no dashboard como *sugestão para confirmação experimental*, não resultado validado.
- Script: `rsm/train_rsm.py`.

### 2.3 — Ajuste Não-Linear de Isotermas e Cinética (`fit/`)

- **Isotermas de equilíbrio**: ajuste direto (não-linearizado) via `scipy.optimize.curve_fit` dos modelos de **Langmuir**, **Freundlich**, **Temkin** e **Redlich-Peterson** sobre os pontos brutos Ce×Qe (36 pontos, só Vagem — Palma só tem os parâmetros já ajustados na planilha original, sem os pontos brutos).
- **Cinética de adsorção**: ajuste de **Pseudo-1ª ordem**, **Pseudo-2ª ordem** e **Elovich** sobre Qₜ×tempo (14 pontos por grupo, cobrindo Palma e Vagem).
- **Funções de erro estatístico** calculadas para cada modelo: χ² (Chi-quadrado), ARE (Erro Relativo Médio), HYBRID (Função de Erro Híbrida) e R²ₙₗ (R² não-linear, calculado diretamente sobre os resíduos do ajuste, sem a distorção da linearização algébrica).
- Achado relevante documentado no dashboard: a linearização clássica de Langmuir (Ce/Qe vs Ce) pode produzir R² artificialmente baixo (ex. 0,03) em blocos com poucos pontos, enquanto o ajuste não-linear direto nos mesmos dados dá R²=0,87 — a estrutura do erro é distorcida pela linearização, não o dado em si.
- Script: `fit/nonlinear_fit.py`.

### 2.4 — Modelagem Molecular: DFT (`dft/`)

- **Nível de teoria**: B3LYP/6-31G(d), via **pyscf** (motor de química quântica nativo em Python) + **ASE**/**RDKit** para construção de geometria — cálculo real, não simulado.
- **Sistemas modelados**: o adsorbato (azul de metileno, cátion) e um monômero representativo da celulose (β-D-glicopiranose) nos três estados de tratamento da biomassa:
  - **In natura**: glicose neutra.
  - **Ácido**: glicose-6-fosfato (éster fosfato formado por H₃PO₄ na hidroxila C6).
  - **Base**: glicose com a hidroxila C6 desprotonada (alcóxido, carga −1) — proxy do efeito de mercerização por NaOH.
- **Propriedades calculadas**: otimização de geometria, energias de HOMO/LUMO, gap de energia, e índices de reatividade da DFT conceitual (dureza química η, maciez S, eletrofilia ω — aproximação de Koopmans).
- **Visualização**: potencial eletrostático molecular (MEP) e isosuperfícies dos orbitais HOMO/LUMO, renderizados interativamente no navegador via **3Dmol.js**, com isovalores calculados a partir do percentil 90 da magnitude de cada cubo (não fixos, pois MEP e orbitais têm escalas muito diferentes).
- **Escopo**: apenas DFT estático — **sem Dinâmica Molecular** (custo computacional incompatível com hardware local; ver notas no dashboard).
- Scripts: `dft/build_structures.py` (geometrias iniciais), `dft/run_dft.py` (cálculo DFT + cubos volumétricos).

---

## 3. Referência dos dados de enriquecimento (estruturas moleculares — DFT)

As estruturas moleculares usadas na análise de DFT **não foram desenhadas manualmente**: as SMILES foram obtidas diretamente da API pública do PubChem (PUG REST), para garantir a estrutura química correta:

| Molécula | Papel na análise | PubChem CID | Fonte |
|---|---|---|---|
| Azul de Metileno (cátion) | Adsorbato | [CID 6099](https://pubchem.ncbi.nlm.nih.gov/compound/6099) | PubChem PUG REST API |
| β-D-Glicopiranose | Proxy da celulose — biomassa *In natura* | [CID 64689](https://pubchem.ncbi.nlm.nih.gov/compound/64689) | PubChem PUG REST API |
| β-D-Glicose 6-fosfato | Proxy da biomassa tratada com **Ácido** (H₃PO₄) | [CID 439427](https://pubchem.ncbi.nlm.nih.gov/compound/439427) | PubChem PUG REST API |

A estrutura do tratamento **Base** (alcóxido no C6) foi derivada diretamente da SMILES da glicopiranose acima (desprotonação da hidroxila primária), não é um composto catalogado separadamente — documentado em `dft/build_structures.py`.

Geometrias 3D iniciais foram construídas com **RDKit** (`ETKDGv3` + pré-otimização MMFF94) antes da otimização final por DFT.

---

## 4. Estrutura do repositório

```
DASHBOARD-VAGEM/
├── index.html              # Dashboard (frontend estático: Chart.js, Plotly.js, 3Dmol.js)
├── db/
│   ├── biosorcao.db         # Banco relacional SQLite (materials/treatments/experiments/...)
│   └── isotherms_static.json# Export estático do banco p/ funcionar sem backend (GitHub Pages)
├── etl/                     # Download do Drive + parsing + validação de schema
├── ml/                      # Pipeline de ML/XAI + backend Flask de predição (uso local)
├── rsm/                     # Pipeline de DoE & RSM
├── fit/                     # Pipeline de ajuste não-linear (isotermas + cinética)
├── dft/                     # Pipeline de DFT (pyscf) + geometrias e cubos volumétricos
└── estetica.md               # Guia de estilo visual usado na construção do dashboard
```

## 5. Rodando localmente

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install flask pandas numpy scikit-learn xgboost shap scipy statsmodels pyscf ase rdkit pyberny google-api-python-client google-auth

# (Re)popular o banco a partir do Google Drive (requer credenciais de service account)
python3 etl/build_database.py

# Servir o dashboard + API de predição
cd ml && python3 app.py
# abrir http://localhost:8765/
```

---

Powered by **William Pinheiro** · [Currículo Lattes](http://lattes.cnpq.br/9111919283952655)

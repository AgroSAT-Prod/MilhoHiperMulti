# MilhoHiperMulti

Repositorio de pesquisa para processamento hiperespectral de milho, extracao de ROIs, geracao de indices espectrais, analises estatisticas, selecao de bandas e modelagem para clorofila, biomassa, dose e tratamento.

O projeto nao esta organizado como pacote Python. Ele funciona como um conjunto de scripts independentes, quase todos executados a partir da raiz do repositorio e dependendo de arquivos CSV/XLSX/TIFF/HDR externos que nem sempre estao versionados.

## Visao geral do fluxo

Fluxo principal observado no codigo:

1. Cubos hiperespectrais e imagens RGB sao lidos a partir de um diretorio local ignorado, normalmente `Dataset/`.
2. Scripts de extracao (`Inicial.py`, `ExtrairVarios.py`, `AlgoritmoROIVariavel*.py`, `ROIVariavelPonderado.py`) produzem tabelas com espectros medios e indices por ROI.
3. Scripts de pre-processamento (`PreProcessamento*.py`) aplicam recorte espectral, SNV, Savitzky-Golay, limpeza e cruzamento com planilhas agronomicas.
4. Scripts analiticos consomem essas tabelas para:
   - ANOVA de uma ou duas vias
   - PCA e Kernel PCA
   - classificacao por dose/tratamento
   - regressao para clorofila e biomassa
   - selecao de bandas por SPA, CARS, greedy, correlacao, VIP, CatBoost, XGBoost e autoencoder
5. Parte dos artefatos gerados foi versionada no repositorio como PNG, CSV, JSON e SQLite.

## Organizacao fisica atual

Estrutura fisica atual do repositorio:

- `/`: todos os scripts, tabelas rastreadas e figuras versionadas ficam na raiz.
- `/.git`: metadados do Git.
- `Dataset/`: pasta esperada por varios scripts, mas ignorada pelo Git.
- `env/`, `venv/`, `.venv/`: ambientes locais ignorados.

Organizacao logica mais util para entender o codigo:

- Extracao de dados hiperespectrais e ROIs
- Pre-processamento e validacao
- Estatistica experimental
- Reducao de dimensionalidade e selecao de bandas
- Modelos para dose/tratamento
- Modelos para clorofila
- Modelos para biomassa
- Artefatos de saida

## Dependencias principais

Bibliotecas encontradas nos scripts:

- `pandas`, `numpy`
- `matplotlib`, `seaborn`
- `scipy`, `statsmodels`
- `scikit-learn`
- `imbalanced-learn`
- `optuna`
- `opencv-python`
- `spectral` / `spectral.io.envi`
- `imageio`
- `catboost`
- `xgboost`
- `tensorflow`, `keras`
- `shap`
- `openpyxl` para leitura de `.xlsx`

## Observacoes importantes antes de rodar

- Muitos scripts usam nomes de arquivos fixos no diretorio atual.
- Alguns scripts usam caminhos absolutos de Windows/OneDrive, entao nao sao portaveis sem ajuste.
- O nome do script nem sempre reflete o modelo atual; alguns arquivos preservam nomes de versoes anteriores.
- O repositorio contem varias variantes experimentais do mesmo pipeline, nao uma unica implementacao oficial.
- Nao ha `requirements.txt`, `pyproject.toml`, testes automatizados ou CLI central.

## Arquivos externos esperados, mas nao versionados

Os seguintes arquivos aparecem no codigo como entradas necessarias em diferentes etapas:

| Arquivo externo | Papel no pipeline | Scripts que referenciam |
| --- | --- | --- |
| `Dataset/` | Pasta base com imagens/cubos hiperespectrais | `.gitignore`, `ANOVACheck.py`, `TestarImagem.py` |
| `PlanilhaFiltrada.xlsx` | Planilha agronomica usada no cruzamento com indices e rotulagem por dose/tratamento | `ANOVA.py`, `ANOVA180.py`, `ANOVATwoWay.py`, `PCAKernelNovo.py`, `PCANovo.py`, `PCATratamentoNovo.py`, `PLSFinalNovo*.py`, `ROIVariavelPonderado.py`, `RandomAutoencoder.py`, `PreProcessamento.py` |
| `PlanilhaFiltrada2.csv` | Variante CSV da planilha agronomica | `ANOVA2.py`, `ANOVACheck.py`, `ANOVATwoWay2.py`, `PreProcessamento1401.py` |
| `spectral_indices_rois.csv` | Tabela de indices/ROIs usada em ANOVA, PCA e classificacao | `ANOVA.py`, `ANOVA180.py`, `ANOVATwoWay.py`, `PCAKernelNovo.py`, `PCANovo.py`, `PCATeste.py`, `PCATratamentoNovo.py`, `PLSFinalNovo*.py`, `RandomAutoencoder.py` |
| `spectral_indices_rois2.csv` | Variante da tabela de indices derivada de `AlgoritmoROIVariavel.py` | `ANOVA2.py`, `ANOVACheck.py`, `ANOVATwoWay2.py`, `AlgoritmoROIVariavel.py` |
| `spectral_indices_rois_final_v3.csv` | Tabela ponderada/final de indices para experimentos balanceados | `ROIVariavelPonderado.py`, `PLSFinalNovoBalanced*.py` |
| `spectral_indices_rois_balanced_final.csv` | Saida da versao WhiteBox do extrator | `AlgoritmoROIVariavelWhiteBox.py` |
| `hiperespectral_ROIs.csv` | Saida bruta de extracao multipla de ROIs | `ExtrairVarios.py`, `PreProcessamentoROI.py` |
| `leituras_hiperspectrais_COMPLETO.csv` | Leitura completa inicial dos cubos | `Inicial.py`, `PreProcessamento1401.py` |
| `DATASET_IA_PROCESSADO.csv` | Base processada usada pela maioria dos modelos classicos | `CARS.py`, `Cat.py`, `Clorofila.py`, `ClorofilaVIP.py`, `Corr.py`, `Greedy.py`, `KBandas.py`, `Modelo*.py`, `PCAKernel.py`, `PLSFinal.py`, `PLSR*.py`, `SPA*.py`, `SVM.py`, `SVR.py` |
| `DATASET_IA_PROCESSADOROI.csv` | Base processada vinda do fluxo de ROI | `PreProcessamento.py`, `PreProcessamentoROI.py` |
| `DATASET_IA_PROCESSADO_JAN2026.csv` | Base intermediaria do fluxo de janeiro de 2026 | `PreProcessamento1401.py` |
| `DATASET_IA_PROCESSADO_JAN2026_VALIDO.csv` | Base filtrada/validada usada no experimento VIP + MAE | `ClorofilaVIPMAE.py` |
| `FINALFINALFINAL.csv` | Base de biomassa usada em modelos e PCA | `BiomassaBaseline.py`, `PCABIomassa.py` |
| `FINAL12.csv` | Base simples de correlacao para biomassa | `BiomassaSimples.py` |
| `VontadeDeMorrer.csv` | Base de teste para experimento de biomassa com Optuna | `BiomassaTeste.py` |
| `*.hdr` e `*-RGB.tiff` | Cubos ENVI e imagens RGB usadas na extracao de ROIs | `ExtrairVarios.py`, `AlgoritmoROIVariavel.py`, `AlgoritmoROIVariavelWhiteBox.py`, `ROIVariavelPonderado.py` |

## Inventario completo dos arquivos versionados

### Metadados e suporte

| Arquivo | O que faz |
| --- | --- |
| `.gitignore` | Ignora ambientes locais, `Dataset/` e a maior parte dos artefatos tabulares/visuais (`*.csv`, `*.png`, `*.jpg`, `*.xlsx`). |
| `.gitkeep` | Arquivo placeholder sem conteudo; provavelmente usado apenas para forcar rastreamento em algum momento. |
| `__.gitkeep_Error.txt` | Log de erro de download/exportacao de um `.gitkeep`, sem papel no pipeline analitico. |
| `db_optuna.sqlite3` | Banco SQLite usado pelos experimentos `PLSFinalNovoTratamentoOptunaDash*.py` para persistir estudos do Optuna. |
| `metricas.json` | Snapshot de metricas, hiperparametros e configuracao de remocao de outliers de um experimento de biomassa. |

### Tabelas rastreadas

| Arquivo | O que faz |
| --- | --- |
| `Biomassa12_12.csv` | Tabela de indices espectrais e variavel-alvo do experimento de biomassa/coleta de `12/12`; usada em `BiomassaBaselinePLSR.py`. |
| `Biomassa14_01.csv` | Tabela de indices e alvo para a coleta de `14/01`, voltada ao fluxo de biomassa. |
| `Biomassa14_01Clorofila.csv` | Variante da coleta de `14/01` com alvo de clorofila; usada em `BiomassaDual.py`, `BiomassaFinal.py` e `BiomassaScatter.py`. |
| `Biomassa23_12.csv` | Tabela de indices e alvo para a coleta de `23/12`, usada como base rastreada de experimento. |
| `Biomassa23_12Clorofila.csv` | Variante da coleta de `23/12` com alvo de clorofila. |

### Figuras rastreadas

| Arquivo | O que faz |
| --- | --- |
| `01_boxplot_clorofila.png` | Boxplot por grupos gerado pelos fluxos `ANOVA.py` e `ANOVA2.py`. |
| `01_boxplot_clorofila_dose180.png` | Boxplot restrito ao recorte de dose 180, gerado por `ANOVA180.py`. |
| `01_interacao_fatorial.png` | Grafico de interacao fatorial do fluxo `ANOVATwoWay2.py`. |
| `01_interacao_tratamento_dose.png` | Grafico de interacao tratamento x dose do fluxo `ANOVATwoWay.py`. |
| `02_boxplot_fatorial.png` | Boxplot fatorial derivado da ANOVA de duas vias em `ANOVATwoWay.py`. |
| `02_heatmap_fatorial.png` | Heatmap do efeito fatorial gerado por `ANOVATwoWay2.py`. |
| `02_heatmap_pvalues.png` | Heatmap de `p-values` da ANOVA de uma via, gerado por `ANOVA.py` e `ANOVA2.py`. |
| `03_heatmap_medias.png` | Heatmap de medias por grupo gerado por `ANOVATwoWay.py`. |
| `03_medias_ic.png` | Grafico de medias com intervalo de confianca da ANOVA de uma via. |
| `03_medias_ic_dose180.png` | Grafico de medias com intervalo de confianca para dose 180. |
| `04_diferencas_significativas.png` | Visualizacao das diferencas significativas pos-hoc da ANOVA de uma via. |
| `analise_outliers.png` | Figura rastreada de inspecao de outliers; o script gerador nao esta explicitamente versionado pelo nome do arquivo. |
| `arvore_decisao_classificacao_multi-classe.png` | Arvore de decisao exportada pelo experimento `PLSFinalNovoPlot.py` para a tarefa multiclasse. |
| `biomassa_resultado.png` | Figura final de biomassa rastreada no repositorio; o nome nao aparece explicitamente nos scripts atuais. |

### Extracao de cubos, mascaras e ROIs

| Arquivo | O que faz |
| --- | --- |
| `Inicial.py` | Faz a leitura inicial de cubos hiperespectrais ENVI, encontra bandas de interesse e exporta leituras completas para `leituras_hiperspectrais_COMPLETO.csv`. |
| `Inicial2.py` | Variante interativa do fluxo inicial; plota NDVI, permite ajuste de mascara e inspecao visual do threshold. |
| `ExtrairVarios.py` | Extrai varias ROIs a partir de cubos ENVI e imagens RGB, com mascara HSV e controle de sobreposicao, gerando `hiperespectral_ROIs.csv`. |
| `AlgoritmoROIVariavel.py` | Pipeline de ROI variavel com mascara HSV, ROIs nao sobrepostas, indices de vegetacao e pre-processamento espectral; produz `spectral_indices_rois2.csv`. |
| `AlgoritmoROIVariavelWhiteBox.py` | Versao white box do extrator de ROI variavel; gera `spectral_indices_rois_balanced_final.csv`. |
| `ROIVariavelPonderado.py` | Extrator de ROI com logica ponderada/recursiva e leitura de doses reais na planilha agronomica; produz `spectral_indices_rois_final_v3.csv`. |
| `TestarImagem.py` | Script utilitario para testar criacao de mascara a partir de uma imagem TIFF RGB. |

### Pre-processamento e validacao

| Arquivo | O que faz |
| --- | --- |
| `PreProcessamento.py` | Aplica recorte de extremidades, SNV e Savitzky-Golay sobre dados espectrais e cruza o resultado com `PlanilhaFiltrada.xlsx`. |
| `PreProcessamentoROI.py` | Mesmo tipo de pre-processamento, mas partindo de `hiperespectral_ROIs.csv`. |
| `PreProcessamento1401.py` | Versao mais completa do pre-processamento para o fluxo de `14/01`, incluindo normalizacao de texto, limpeza numerica, remocao de outliers por IQR e grafico `processamento_espectral_4steps.png`. |
| `ANOVACheck.py` | Valida o cruzamento entre `spectral_indices_rois2.csv` e `PlanilhaFiltrada2.csv`, ajudando a depurar correspondencias e consistencia dos dados. |
| `Corr.py` | Analisa correlacao entre bandas/variaveis da base `DATASET_IA_PROCESSADO.csv`; funciona como exploracao estatistica de apoio. |

### ANOVA e estatistica experimental

| Arquivo | O que faz |
| --- | --- |
| `ANOVA.py` | Executa ANOVA de uma via sobre `spectral_indices_rois.csv` cruzado com `PlanilhaFiltrada.xlsx`, incluindo testes de pressupostos, pos-hoc e geracao de quatro figuras. |
| `ANOVA180.py` | Variante da ANOVA de uma via focada no subconjunto de dose 180, gerando boxplot e grafico de medias com IC. |
| `ANOVA2.py` | Evolucao do fluxo `ANOVA.py` usando `spectral_indices_rois2.csv` e `PlanilhaFiltrada2.csv`, com limpeza adicional de texto e numeros. |
| `ANOVATwoWay.py` | Executa ANOVA de duas vias com interacao tratamento x dose e gera graficos de interacao, boxplot fatorial e heatmap de medias. |
| `ANOVATwoWay2.py` | Variante da ANOVA de duas vias para a base `spectral_indices_rois2.csv`, com normalizacao textual e foco no efeito fatorial. |

### Modelos classicos, selecao de bandas e aprendizado supervisionado

| Arquivo | O que faz |
| --- | --- |
| `Modelo.py` | Classificador base sobre `DATASET_IA_PROCESSADO.csv`, com avaliacao, importancia de variaveis e assinatura media por classe. |
| `ModeloBag.py` | Variante do `Modelo.py` baseada em ensemble do tipo bagging. |
| `ModeloCPearson.py` | Combina analise de correlacao de Pearson com avaliacao preditiva e importancia de atributos. |
| `ModeloInteresse.py` | Repete o fluxo de classificacao usando apenas bandas/colunas de interesse previamente selecionadas. |
| `ModeloInteresseLOO.py` | Variante com validacao leave-one-out sobre bandas de interesse. |
| `SVM.py` | Classificacao com SVM, com visualizacao de coeficientes e assinatura media por classe. |
| `SVR.py` | Regressao com SVR para alvo continuo, incluindo analise de coeficientes e assinatura media. |
| `PLSR.py` | Regressao PLSR classica sobre a base processada, com metricas e graficos de coeficientes. |
| `PLSR2.py` | Variante hibrida que combina tratamento espectral e modelo de regressao/Random Forest para comparacao. |
| `SPA.py` | Implementa o seletor SPA (Successive Projections Algorithm) e avalia subconjuntos de bandas com modelo supervisionado. |
| `SPA2.py` | Versao do SPA com limpeza de bandas e visualizacao do espectro selecionado. |
| `SPA3.py` | Versao mais avancada do SPA com balanceamento e validacao biologica das bandas escolhidas. |
| `SPAFinal.py` | Consolida o fluxo SPA com correlacao espectral, boxplots, importancia de features e ranking de bandas ligadas a nitrogenio. |
| `Greedy.py` | Selecao gulosa de bandas espectralmente distantes, seguida de avaliacao do modelo e graficos de apoio. |
| `KBandas.py` | Seleciona `k` bandas, gera frames para GIF e ajuda a visualizar a evolucao do processo de escolha. |
| `CARS.py` | Implementa o seletor CARS em uma classe propria e avalia as bandas escolhidas em um modelo supervisionado. |
| `Cat.py` | Classificacao com CatBoost sobre `DATASET_IA_PROCESSADO.csv`, incluindo importancia e assinatura media por classe. |
| `Clorofila.py` | Busca combinacoes de bandas que maximizem `R2` para estimar clorofila, gerando evolucao, scatter e ranking das melhores combinacoes. |
| `ClorofilaVIP.py` | Expande `Clorofila.py` com calculo de VIP para ranquear bandas relevantes. |
| `ClorofilaVIPMAE.py` | Versao mais completa do fluxo de clorofila, com VIP, MAE, coeficientes, violin plots e exportacao de tabelas de metricas. |
| `RandomAutoencoder.py` | Explora selecao de bandas com autoencoder compacto, balanceamento, CatBoost e SHAP, gerando matrizes e trade-off de bandas. |

### PCA, Kernel PCA e analise de estrutura

| Arquivo | O que faz |
| --- | --- |
| `PCAKernel.py` | Aplica Kernel PCA e analise linear auxiliar sobre `DATASET_IA_PROCESSADO.csv`, com foco em separacao de grupos e importancia das bandas. |
| `PCAKernelNovo.py` | Variante de Kernel PCA cruzada com `PlanilhaFiltrada.xlsx`, com filtro por doses e exportacao de graficos 3D e de contribuicao espectral. |
| `PCANovo.py` | Executa PCA classico sobre `spectral_indices_rois.csv` + planilha agronomica, gerando variancia explicada, clusters e contribuicao de bandas. |
| `PCATeste.py` | Variante de PCA com biplot adicional para inspecao visual dos clusters. |
| `PCATratamentoNovo.py` | Usa PCA como etapa de representacao para comparar classificacao multiclasse e binaria por tratamento/dose. |
| `PCABIomassa.py` | Aplica PCA e biplot a base de biomassa `FINALFINALFINAL.csv`. |

### Pipelines PLSFinalNovo e variantes por dose/tratamento

| Arquivo | O que faz |
| --- | --- |
| `PLSFinal.py` | Pipeline final de bandas selecionadas para nitrogenio/classe, com seletor SPA, seletor PLS, correlacao espectral e classificacao supervisionada. |
| `PLSFinalNovo.py` | Benchmark de classificacao multiclasse e binaria usando `spectral_indices_rois.csv` e `PlanilhaFiltrada.xlsx`. |
| `PLSFinalNovoBalanced.py` | Mesmo benchmarking, mas usando a base balanceada `spectral_indices_rois_final_v3.csv`. |
| `PLSFinalNovoBalancedCat.py` | Variante balanceada com CatBoost e graficos de importancia para cenarios binario e multiclasse. |
| `PLSFinalNovoBalancedCatGrid.py` | Variante com CatBoost voltada a matriz final otimizada e visualizacao por t-SNE. |
| `PLSFinalNovoBalancedXG.py` | Variante balanceada com XGBoost e graficos de importancia especificos. |
| `PLSFinalNovoOptuna.py` | Variante com otimizacao de hiperparametros por Optuna sobre o fluxo de classificacao. |
| `PLSFinalNovoPlot.py` | Variante que salva graficos comparativos e exporta arvore de decisao, incluindo `arvore_decisao_classificacao_multi-classe.png`. |
| `PLSFinalNovoTratamento.py` | Ajusta o pipeline para foco explicito em tratamento, mantendo comparacao binaria e multiclasse. |
| `PLSFinalNovoTratamentoOptuna.py` | Adiciona otimizacao por Optuna ao fluxo focado em tratamento. |
| `PLSFinalNovoTratamentoOptunaDash.py` | Variante voltada a uso continuo com armazenamento de estudos no SQLite `db_optuna.sqlite3`. |
| `PLSFinalNovoTratamentoOptunaDashMore.py` | Extensao do fluxo Dash/Optuna com mais controle sobre parsing textual e execucao do estudo. |
| `PLSFinalNovoTratamentoPorc.py` | Variante por tratamento usando regras/agrupamentos percentuais nas classes. |

### Biomassa e regressao de alvo continuo

| Arquivo | O que faz |
| --- | --- |
| `BiomassaBaseline.py` | Baseline de biomassa com Random Forest e validacao leave-one-out sobre uma base externa em caminho absoluto. |
| `BiomassaBaselinePLSR.py` | Baseline simples de regressao para `Biomassa12_12.csv`, com exportacao de CSV e grafico final. |
| `BiomassaDual.py` | Regressao PLSR para `Biomassa14_01Clorofila.csv`, com escolha do numero de componentes e saidas de desempenho. |
| `BiomassaFinal.py` | Fluxo mais completo de biomassa/clorofila com remocao iterativa de outliers, Optuna e exportacao de resultados finais. |
| `BiomassaScatter.py` | Estende `BiomassaFinal.py` com graficos de dispersao por indice versus alvo. |
| `BiomassaSimples.py` | Gera correlacoes e graficos exploratorios simples a partir de uma base externa de biomassa. |
| `BiomassaTeste.py` | Experimento alternativo de biomassa com Random Forest otimizado por Optuna em uma base externa. |

## Relacao entre familias de scripts

Os grupos abaixo representam a progressao mais clara encontrada no repositorio:

- `Inicial.py` -> `ExtrairVarios.py` / `AlgoritmoROIVariavel.py` / `ROIVariavelPonderado.py`
- `AlgoritmoROIVariavel*.py` -> `spectral_indices_rois*.csv`
- `PreProcessamento*.py` -> `DATASET_IA_PROCESSADO*.csv`
- `ANOVA*.py` e `ANOVATwoWay*.py` -> analise estatistica dos indices
- `PCA*.py`, `PCAKernel*.py` -> estrutura e separacao dos grupos
- `Modelo*.py`, `SVM.py`, `SVR.py`, `PLSR*.py`, `SPA*.py`, `CARS.py`, `Greedy.py`, `KBandas.py`, `Cat.py` -> modelagem e selecao de bandas
- `Clorofila*.py` -> otimizacao de bandas para clorofila
- `Biomassa*.py` -> regressao em bases resumidas de biomassa/clorofila

## Limitacoes atuais do repositorio

- Estrutura totalmente flat, sem separacao entre codigo, dados e resultados.
- Forte duplicacao de funcoes utilitarias entre arquivos.
- Muitos nomes de datasets e caminhos absolutos estao hardcoded.
- Nao existe documentacao original do pipeline nem convencao unica de nomenclatura.
- Varios artefatos rastreados sao saidas de experimentos, nao entradas primarias.

## Sugestao de reorganizacao futura

Caso o projeto evolua, a organizacao natural seria:

- `data/raw/`: cubos, TIFFs, planilhas originais
- `data/interim/`: `spectral_indices_rois*.csv`, `leituras_*.csv`
- `data/processed/`: `DATASET_IA_PROCESSADO*.csv`
- `src/extraction/`: scripts de ROI e leitura hiperespectral
- `src/preprocessing/`: limpeza e transformacao
- `src/stats/`: ANOVA e validacao
- `src/models/`: classificacao, regressao e selecao de bandas
- `results/figures/`: PNGs
- `results/tables/`: CSV/JSON/SQLite de saida

Hoje, porem, essa separacao ainda nao existe no repositorio versionado.

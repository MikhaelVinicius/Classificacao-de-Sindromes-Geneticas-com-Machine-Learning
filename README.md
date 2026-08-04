# Apollo Solutions - Machine Learning Practical Test

## Descrição do Projeto

Este projeto foi desenvolvido como parte do teste prático para a vaga de Machine Learning Developer na Apollo Solutions. O objetivo principal é analisar embeddings extraídos de imagens (vetores de 320 dimensões) para classificar síndromes genéticas.

O pipeline completo abrange desde o processamento de dados aninhados em um arquivo `.pickle` até a Análise Exploratória de Dados (EDA), visualização com t-SNE, e a implementação de um classificador K-Nearest Neighbors (KNN) customizado. O modelo foi avaliado através de 10-fold cross-validation, comparando as métricas de distância Euclidiana e de Cosseno, e calculando métricas chaves como AUC, F1-Score e Top-3 Accuracy.

## Estrutura do Projeto

O projeto está organizado da seguinte forma:

```text
├── data/
│   └── mini_gm_public_v0.1.p        # Dataset original fornecido (Pickle)
├── plots/                           # Diretório gerado automaticamente com os resultados
│   ├── k_optimization_*.png         # Gráficos da otimização do K (Acurácia, F1, AUC, Top-3)
│   ├── performance_summary_table.md # Tabela final de resumo das métricas
│   ├── roc_curve_knn_*.png          # Curvas ROC (Euclidiana e Cosseno)
│   └── tsne_embeddings_plot.png     # Visualização t-SNE dos embeddings
├── src/                             # Scripts Python com o código fonte
│   ├── data_loader.py               # Módulo de carregamento e inspeção de dados
│   ├── data_preprocessor.py         # Módulo de achatamento (flatten) e integridade
│   ├── data_visualizer.py           # Módulo de redução de dimensionalidade (t-SNE)
│   ├── eda.py                       # Módulo de Análise Exploratória de Dados
│   ├── knn_classifier.py            # Implementação do KNN customizado
│   ├── main.py                      # Arquivo principal que executa o pipeline
│   ├── metrics_calculator.py        # Implementação do cálculo manual de métricas (ROC, F1, Top-k)
│   ├── model_evaluator.py           # Validação cruzada (10-fold) e busca do k ótimo
│   └── results_visualizer.py        # Geração de gráficos e tabelas finais
├── requirements.txt                 # Dependências do projeto
└── README.md                        # Documentação do projeto
```

## Pré-requisitos e Instalação

Certifique-se de ter o Python 3.8 ou superior instalado.

Para instalar as dependências necessárias, execute o seguinte comando no terminal na raiz do projeto (recomendamos o uso de um ambiente virtual - `venv`):

```bash
pip install -r requirements.txt
```

**Nota:** O pacote `tabulate` está incluído nas dependências para garantir a correta formatação da tabela Markdown gerada no terminal.

## Como Executar

O projeto foi desenvolvido para rodar como um pipeline completo através de um único script principal, cumprindo o requisito de não utilizar Jupyter Notebooks.

A partir do diretório raiz do projeto, execute:

```bash
python src/main.py
```

## Fluxo de Execução

Ao executar o comando acima, o pipeline fará o seguinte automaticamente:

1. **Carregamento:** Lerá o arquivo `.pickle` na pasta `data/`.
2. **Pré-processamento e EDA:** Achatamento dos embeddings e impressão de estatísticas de desbalanceamento de classes no terminal.
3. **Visualização:** Geração do gráfico t-SNE 2D.
4. **Modelagem:** Treinamento do KNN e execução de 10-Fold Cross-Validation para hiperparametrização (buscando o melhor `k` de 1 a 15, otimizando pelo F1-Score) utilizando as distâncias Euclidiana e de Cosseno.
5. **Resultados:** Salvamento das curvas ROC, gráficos de otimização e geração da tabela comparativa final na pasta `plots`.
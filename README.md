#  Projeto 5 - Pipeline de Reconhecimento de Entidades (NER) e Clusterização Semântica

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![spaCy](https://img.shields.io/badge/spaCy-09A3D5?style=for-the-badge&logo=spaCy&logoColor=white)
![Transformers](https://img.shields.io/badge/HuggingFace-F9AB00?style=for-the-badge&logo=huggingface&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)

##  Objetivo do Projeto
Este projeto implementa um pipeline avançado de Processamento de Linguagem Natural (PLN) para analisar textos científicos e de projetos de inovação. O objetivo principal é **extrair informações cruciais (Entidades Nomeadas) utilizando a biblioteca SpaCy** e, em seguida, **agrupar essas entidades por semelhança semântica utilizando modelos de linguagem Transformers (Embeddings) e técnicas de clusterização (K-Means e t-SNE)**.

O código foi projetado com foco em **otimização de processamento computacional**, **gestão de tamanho de rede neural** e **tolerância a falhas em ambientes de nuvem (Kaggle/Colab)**.

---

##  Sobre o Dataset
* **Origem:** Base de dados hospedada no Kaggle (`georgekurokijr/dl-2024`).
* **Arquivo:** `dadosTextosCientificos.tsv` (Tab-Separated Values).
* **Atributo Alvo:** A coluna principal analisada é a `Descricao_pública`, que contém os resumos técnicos de projetos de pesquisa e inovação tecnológica.
* **Tolerância a Falhas:** O código possui um mecanismo de *fallback* (Célula 2). Caso o dataset não seja encontrado no diretório especificado ou o Kaggle esteja sem internet, o sistema injeta automaticamente um volume de textos fictícios (`Mock Data`) para garantir que o pipeline seja executado e validado até o fim.

---

##  Ontologia e Entidades Nomeadas (NER)
Para garantir um padrão profissional de extração, superando entidades genéricas, o modelo foi ajustado (Fine-Tuning) para reconhecer uma ontologia específica focada no ecossistema de inovação e pesquisa. As classes são:

* **`ORGANIZACAO`**: Empresas, universidades e instituições (ex: Suzano, Klabin).
* **`TECNOLOGIA`**: Ferramentas e métodos técnicos (ex: inteligência artificial, aprendizado de máquina).
* **`AREA_PESQUISA`**: Campos de estudo (ex: agricultura de precisão, bioeconomia).
* **`PRODUTO`**: Materiais ou resultados de desenvolvimento (ex: Açúcar de frutas).
* **`METRICA_AMBIENTAL`**: Termos relacionados à sustentabilidade (ex: sequestro de carbono).
* **`LOCAL`**: Regiões geográficas e países.

---

##  Arquitetura do Código e Funções Principais

O notebook está estruturado em 5 etapas lógicas e independentes:

### 1. Setup e Importações
Instalação silenciosa de dependências e download do modelo base do idioma português (`pt_core_news_sm`). 

### 2. Ingestão e Tratamento de Dados
Carregamento do arquivo `.tsv` usando `pandas`. Trata valores nulos e extrai os textos para uma lista pura de Python, otimizando o consumo de RAM.

### 3. Treinamento Otimizado (SpaCy) - *Destaque Técnico*
* **Fine-tuning Base:** Adiciona os rótulos personalizados ao pipeline `ner` nativo.
* **Economia de Processamento (`disable_pipes`):** Implementamos a função `nlp.disable_pipes(*unaffected_pipes)` para congelar temporariamente os componentes sintáticos do modelo (como *tagger* e *parser*). Isso concentra 100% do poder computacional da CPU/GPU apenas no aprendizado de entidades, reduzindo drasticamente o custo de processamento.
* **Tratamento Offline:** Caso o ambiente bloqueie o download do modelo base, o código inicializa de forma tolerante a falhas usando `spacy.blank("pt")`.

### 4. Inferência e Extração
Aplica o modelo treinado sobre os primeiros 150 projetos (amostragem para evitar poluição visual), aplica regras de limpeza (remoção de pontuação residual) e cria uma lista de entidades exclusivas (Set).

### 5. Embeddings Semânticos e Clusterização (Transformers)
* **Cálculo da Rede:** O código varre a arquitetura do Transformer usando `sum(p.numel() for p in modelo.parameters())`, atestando que a rede possui aproximadamente **117 milhões de parâmetros**, o que a qualifica como uma rede eficiente (MiniLM) e não superdimensionada.
* **Modelo Escolhido:** Utilizamos o `paraphrase-multilingual-MiniLM-L12-v2`. Ao contrário de modelos BERT base (que geram alertas de *unexpected weights* em embeddings), a família *SentenceTransformers* é nativamente treinada para clusterização semântica e gera representações vetoriais de altíssima qualidade com baixo custo.
* **Técnicas Matemáticas:** Aplicação de **K-Means** para segmentação das entidades e **t-SNE** (T-distributed Stochastic Neighbor Embedding) para redução de dimensionalidade e visualização bidimensional.
* **Visualização Limpa:** A plotagem via `matplotlib` e `seaborn` usa um sorteio probabilístico (`np.random.choice`) para plotar apenas o nome de 30 amostras no gráfico, impedindo a sobreposição de textos e mantendo a legibilidade visual profissional.

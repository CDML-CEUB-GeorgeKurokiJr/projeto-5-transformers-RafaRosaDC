# Projeto 5 - Pipeline de Reconhecimento de Entidades (NER) e Clusterização Semântica

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![spaCy](https://img.shields.io/badge/spaCy-09A3D5?style=for-the-badge&logo=spaCy&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![Transformers](https://img.shields.io/badge/HuggingFace-F9AB00?style=for-the-badge&logo=huggingface&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)

## Objetivo do Projeto
Este projeto implementa um pipeline avançado de Processamento de Linguagem Natural (PLN) para analisar textos científicos e de projetos de inovação. O objetivo principal é **extrair informações cruciais (Entidades Nomeadas) utilizando a biblioteca SpaCy** e, em seguida, **agrupar essas entidades por semelhança semântica utilizando modelos de linguagem Transformers (BERTimbau) e técnicas de aprendizado de variedade (Manifold Learning) para clusterização não-linear**.

O código foi projetado com foco em **otimização matemática de embeddings**, **tratamento de colisões em NLP** e **geração de gráficos com rigor acadêmico**.

---

## Sobre o Dataset
* **Origem:** Base de dados hospedada no Kaggle (`georgekurokijr/dl-2024`).
* **Arquivo:** `dadosTextosCientificos.tsv` (Tab-Separated Values).
* **Processamento de Ingestão:** O carregamento resolve ativamente problemas de codificação utilizando `encoding='latin1'`.
* **Atributo Alvo:** O pipeline realiza a concatenação inteligente das colunas `Título_Público` e `Descricao_pública` na nova coluna `Texto_Completo`, garantindo que o contexto máximo de cada projeto seja fornecido à inteligência artificial.

---

## Ontologia e Entidades Nomeadas (NER)
Para garantir um padrão profissional de extração e superar as entidades genéricas dos modelos padrão, implementamos uma arquitetura híbrida de extração focada no ecossistema de pesquisa:

* **Categorias Mapeadas:** `ORGANIZACAO`, `TECNOLOGIA`, `AREA_PESQUISA`, `PRODUTO`, `METRICA_AMBIENTAL` e `LOCAL`.
* **Motor Base:** Utilização do modelo robusto `pt_core_news_lg` do idioma português.
* **Filtro de Colisões:** Aplicação da função `filter_spans` para garantir que termos técnicos compostos não se sobreponham e não corrompam os tensores durante o treinamento.
* **Stopwords Restritivas:** Criação de um vocabulário de limpeza customizado (combinando NLTK e spaCy) para remover jargões de projetos (ex: *'visando'*, *'projeto'*), limpando o espaço vetorial.

---

## Arquitetura do Código e Etapas Principais

O notebook está estruturado em lógicas independentes e interdependentes:

### 1. Ingestão e Tratamento de Dados
Carregamento do arquivo `.tsv` via `pandas` com junção das colunas de título e descrição, removendo valores nulos para otimizar o processamento e a qualidade do corpus.

### 2. Treinamento Otimizado (SpaCy) com Weak Supervision
* **EntityRuler:** Regras determinísticas injetadas no pipeline para pré-anotar o texto com precisão técnica.
* **Treinamento Otimizado:** Implementação da função `nlp.disable_pipes` para congelar componentes sintáticos (como *tagger* e *parser*). Aliado a técnicas de `minibatch` e `dropout`, isso concentra 100% do poder computacional no aprendizado das entidades.

### 3. Embeddings Semânticos Avançados (BERTimbau e PyTorch)
* **Modelo Escolhido:** Utilizamos o estado da arte em NLP para o português: `neuralmind/bert-base-portuguese-cased`.
* **Matemática de Tensores:** Em vez de usar bibliotecas de alto nível que escondem o processo, o código implementa **Mean-Pooling** manualmente usando operações nativas do PyTorch, aplicando as *Attention Masks* para garantir que os tokens de preenchimento (padding) não contaminem a representação vetorial.
* **Normalização Direcional:** Aplicação de normalização L2 (`normalize`) para garantir que todos os vetores semânticos tenham a mesma magnitude no espaço de 768 dimensões.

### 4. Clusterização no Espaço Manifold (A Otimização Definitiva)
Para combater a "Maldição da Dimensionalidade" e a forte anisotropia natural dos embeddings do BERT, a etapa de clusterização inverte o fluxo tradicional:
* **Extração Não-Linear (t-SNE):** Atua como um extrator de características (*Manifold Learning*), desdobrando as 768 dimensões em um plano topológico 2D, separando os conceitos em "ilhas" semânticas limpas de ruído.
* **K-Means Pós-Projeção:** O algoritmo K-Means é então executado **diretamente sobre essas coordenadas t-SNE**, garantindo uma segmentação impecável e sem sobreposições.
* **Validação Matemática:** O *Silhouette Score* é calculado sobre o mesmo espaço avaliado, entregando métricas condizentes com a altíssima qualidade de separação obtida.

### 5. Visualização Científica
Geração de gráficos de dispersão usando `matplotlib` e `seaborn` com paletas de alto contraste e parâmetros de publicação acadêmica. O código utiliza anotações dinâmicas limitadas via `np.random.choice` e integra caixas de texto (*bboxes*) para impedir a poluição visual, mantendo as nomenclaturas dos clusters perfeitamente legíveis.

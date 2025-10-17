# Relatório Técnico - Desafio EWork

Este documento detalha as decisões de arquitetura, trade-offs e os próximos passos para a solução desenvolvida, que consiste em um pipeline de ETL para qualidade de dados e uma API de RAG para busca semântica.

## 1. Decisões Técnicas e de Design

A arquitetura do projeto foi guiada por princípios de robustez, manutenibilidade e portabilidade, utilizando uma stack de ferramentas moderna e eficiente.

### Pipeline de Dados (ETL)

* **Framework de Dados: Polars**
    A escolha do Polars foi estratégica por sua performance superior em operações de I/O e manipulação de DataFrames. Sua API expressiva e o motor de execução *lazy* permitiram um código mais limpo e otimizado, ideal para lidar com as transformações e validações de forma eficiente.

* **Validação Declarativa com Pandera**
    Em vez de lógica de validação imperativa (com `if/else`), foi adotado o Pandera para definir "contratos de dados" declarativos. Esta abordagem separa as regras de negócio do código de transformação, tornando o pipeline mais robusto, fácil de auditar e de manter.

* **Política de Quarentena e Logs Estruturados**
    A política de **quarentenar registros inválidos**, em vez de descartá-los, foi uma decisão chave para garantir a confiabilidade e a auditabilidade do pipeline, sem perda de informação. A integração do **Loguru** para logs estruturados substituiu os `print()`, fornecendo uma visibilidade clara de cada etapa, essencial para o debug e monitoramento em produção.

### Mini-RAG (Retrieval-Augmented Generation)

* **Arquitetura Local-First e Portátil**
    A solução de RAG foi construída para ser totalmente portátil e de custo-zero, utilizando uma stack open-source que roda localmente, empacotada em Docker:
    * **LLM:** **Ollama** com o modelo `llama3.1`, demonstrando a capacidade de integrar LLMs de forma agnóstica.
    * **Embeddings & Vector Store:** O modelo `sentence-transformers/all-MiniLM-L6-v2` e a biblioteca **FAISS** foram escolhidos pelo excelente equilíbrio entre performance, leveza e qualidade, ideal para uma POC robusta.
    * **API:** O **FastAPI** foi utilizado pela sua alta performance, tipagem nativa e a geração automática de documentação, que melhora drasticamente a Developer Experience (DX).

### Empacotamento e Execução com Docker

* O projeto foi totalmente containerizado com **Docker** e **Docker Compose**, garantindo um ambiente de execução 100% reprodutível e isolado. Esta decisão abstrai toda a complexidade de setup e dependências, permitindo que qualquer desenvolvedor suba a aplicação com um único comando (`docker-compose up`). A imagem da API foi otimizada para ser leve, separando dependências de produção e desenvolvimento, uma prática essencial para deploy.

## 2. Trade-offs e Aprendizados de Depuração

* **Trade-off: Imputação vs. Quarentena:** A escolha de quarentenar dados inválidos priorizou a **confiabilidade dos dados** sobre a retenção de 100% dos registros. Embora a imputação pudesse "salvar" mais linhas, ela introduziria o risco de poluir o dataset com dados sintéticos.

* **Trade-off: Simplicidade do RAG:** A pipeline implementada é um baseline funcional (Retrieve & Generate). Não foram adicionadas camadas de complexidade como *reranking*, para priorizar a entrega de uma solução end-to-end robusta e funcional dentro do escopo do desafio.

* **Aprendizado de Depuração Profunda (Networking em Docker/Linux):** Um desafio significativo foi estabelecer a comunicação entre o contêiner da API e o serviço do Ollama rodando no host Linux. A depuração revelou que múltiplas camadas precisavam ser corrigidas:
    1.  O `localhost` dentro de um contêiner não é a máquina host.
    2.  O `host.docker.internal` não funciona por padrão no Linux, exigindo o uso do IP da bridge (`172.17.0.1`).
    3.  A causa raiz final era a configuração do serviço **`systemd`** do Ollama, que o forçava a escutar apenas em `127.0.0.1`. A solução definitiva envolveu a edição direta do arquivo de serviço (`/etc/systemd/system/ollama.service`) para definir `OLLAMA_HOST=0.0.0.0`.
    Essa experiência reforçou a importância de compreender a stack de rede subjacente em ambientes containerizados.

## 3. Próximos Passos e Evolução

A base atual do projeto é sólida e pronta para evoluir. Os próximos passos lógicos seriam:

1.  **Orquestração e Agendamento:** Integrar o pipeline de ETL a um orquestrador como **Airflow** ou **Dagster** para agendar execuções, monitorar o status e gerenciar retentativas de forma automática.

2.  **Melhoria e Avaliação do RAG:**
    * Implementar uma etapa de **reranking** (ex: com um Cross-Encoder) para melhorar a relevância do contexto enviado ao LLM.
    * Construir uma suíte de avaliação automatizada com frameworks como **RAGAs** para medir objetivamente a qualidade das respostas (ex: *faithfulness*, *answer relevancy*).

3.  **Hardening e Deploy:**
    * Mover a imagem Docker para um registro de contêineres na nuvem (ex: AWS ECR).
    * Fazer o deploy da API em um serviço como **AWS ECS** ou **Fargate**, com auto-scaling e logging centralizado.
    * Adicionar autenticação e rate limiting à API.

4.  **Versionamento de Dados e Modelos:** Utilizar ferramentas como **DVC (Data Version Control)** para versionar os datasets processados, os índices FAISS e os modelos de embedding, garantindo a reprodutibilidade completa dos experimentos e do pipeline.
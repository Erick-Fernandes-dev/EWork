# Dockerfile

# Passo 1: Imagem Base
FROM python:3.12-slim

# Passo 2: Definir o diretório de trabalho dentro do contêiner
WORKDIR /app

# Passo 3: Instalar o Poetry
RUN pip install poetry
RUN poetry config virtualenvs.create false

# Passo 4: Copiar e Instalar Dependências (Camada de Cache)
COPY pyproject.toml poetry.lock ./

# CORREÇÃO AQUI: Trocamos o comando antigo '--no-dev' pelo moderno '--without dev'
RUN poetry install --no-root --without dev

# Passo 5: Copiar o Código-Fonte
COPY . .

# Passo 6: Comando Padrão
CMD ["poetry", "run", "uvicorn", "src.rag.api:app", "--host", "0.0.0.0", "--port", "8003"]
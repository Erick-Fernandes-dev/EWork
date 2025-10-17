# Desafio EWork

Alguns requisitoss iniciais:

- Ter o Python 3.11+
- Instalar o poetry
- Instalar o ollama

Alguns comando que usei no ollama:

```python
ollama pull llama3.1
ollama list
ollama run llama3.1
```

![image.png](Desafio%20EWork%2028f2e2f65d7e813689cdf9e52ba109d0/image.png)

```python
mkdir -p ework-desafio/raw ework-desafio/data ework-desafio/docs ework-desafio/schemas ework-desafio/src/etl ework-desafio/src/rag ework-desafio/tests ework-desafio/eval
cd ework-desafio

```

Inicialize o projeto com Poetry:

```python
poetry init --name "ework-desafio" --description "Desafio Técnico EWork" -n

poetry add "polars[pyarrow]" pandera pydantic fastapi uvicorn "python-dotenv[cli]" \
"sentence-transformers" "faiss-cpu==1.8.0" pypdf requests \
"pytest" "ruff" "black" "mypy"
```

Crie os arquivos iniciais:

```python
touch src/etl/__init__.py src/rag/__init__.py src/__init__.py
touch src/etl/run.py src/etl/validators.py
touch src/rag/api.py src/rag/ingest.py src/rag/retriever.py
touch tests/test_etl.py tests/test_rag.py
touch Makefile README.md REPORT.md .env .pre-commit-config.yaml
```

Vamos, configurar **o `Makefile`:** Cole o seguinte conteúdo no arquivo `Makefile:`

```makefile
# Makefile
.PHONY: all setup lint format test run-etl run-rag-ingest run-rag-api clean

VENV_ACTIVATE = $(shell poetry env info --path)/bin/activate

all: setup lint test

setup:
	@echo "--> Installing dependencies with Poetry..."
	poetry install

lint:
	@echo "--> Running linter (ruff)..."
	poetry run ruff check .
	@echo "--> Checking formatting (black)..."
	poetry run black --check .
	@echo "--> Checking types (mypy)..."
	poetry run mypy .

format:
	@echo "--> Formatting with ruff and black..."
	poetry run ruff check . --fix
	poetry run black .

test:
	@echo "--> Running tests..."
	poetry run pytest -q

run-etl:
	@echo "--> Running data ETL pipeline..."
	poetry run python -m src.etl.run

run-rag-ingest:
	@echo "--> Running RAG ingestion (PDF to VectorDB)..."
	poetry run python -m src.rag.ingest

run-rag-api:
	@echo "--> Starting RAG API server at http://localhost:8003"
	poetry run uvicorn src.rag.api:app --host 0.0.0.0 --port 8003 --reload

clean:
	@echo "--> Cleaning up generated files..."
	rm -rf data/* silver/* docs/*.pdf db/*
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
```

Criei um Script Python para organizar os dados do 1 arquivo:

```makefile
import csv
import os

# --- Dados a serem escritos no arquivo CSV ---
# Note que os problemas intencionais foram mantidos exatamente como no exemplo.
data = [
    ['product_id', 'sku', 'model', 'category', 'weight_grams', 'dimensions_mm', 'vendor_code', 'launch_date', 'msrp_usd'],
    ['1001', 'AB-001', 'Alpha-X', 'Router', '950', '220x120x45', 'V-77', '2023-11-15', '129.90'],
    ['1002', 'AB-002', 'Alpha-X Pro', 'Router', '', '220x120x45', 'V-77', '2023/13/01', '159,90'],
    ['1003', 'ZX-900', '', 'Switch', '1800', '440x300x44', 'V-12', '2022-05-07', '499.00'],
    ['1004', 'ZZ-001', 'OmegaCam', 'Camera', '650', '90x60x', 'V-77', '2021-02-29', '249.00'],
    ['', 'AB-003', 'Alpha-Mini', 'Router', '420', '120x80x30', 'V-77', '2024-03-12', '99.00']
]

# --- Nome do arquivo de saída ---
file_name = 'raw_products.csv'

# --- Lógica para criar e escrever no arquivo ---
try:
    # A instrução 'with open' garante que o arquivo seja fechado corretamente
    # newline='' evita a criação de linhas em branco extras no Windows
    with open(file_name, 'w', newline='', encoding='utf-8') as csvfile:
        # Cria um objeto 'writer' para escrever no formato CSV
        csv_writer = csv.writer(csvfile)
        
        # Escreve todas as linhas da lista 'data' no arquivo
        csv_writer.writerows(data)
        
    print(f"✅ Arquivo '{os.path.abspath(file_name)}' criado com sucesso!")

except IOError as e:
    print(f"❌ Erro ao escrever o arquivo: {e}")
```

```python
pytyhon3 criando_csv_row_product.py
```

![image.png](Desafio%20EWork%2028f2e2f65d7e813689cdf9e52ba109d0/image%201.png)

Ele cria um arquivo em CSV com os dados organizados

![image.png](Desafio%20EWork%2028f2e2f65d7e813689cdf9e52ba109d0/image%202.png)

**`raw/inventory.parquet`:** Crie um pequeno script para gerar este arquivo. Chame-o de `create_raw_data.py` na raiz do projeto.

```python
# create_raw_data.py
import polars as pl
from datetime import datetime
import os

def create_inventory_data():
    if not os.path.exists("raw"):
        os.makedirs("raw")

    df = pl.DataFrame({
        "product_id": [1001, 1002, 1003, 9999, 1001, 1002], # 9999 é um ID inexistente
        "warehouse": ["WH-A", "WH-A", "WH-B", "WH-A", "WH-B", "WH-C"],
        "on_hand": [150, 80, 200, 10, -5, 120], # -5 é um valor inválido
        "min_stock": [50, 50, 100, 5, 20, 40],
        "last_counted_at": [
            datetime(2024, 1, 10), datetime(2024, 1, 12),
            datetime(2024, 1, 5), datetime(2024, 1, 8),
            datetime(2024, 1, 15), datetime(2024, 1, 9)
        ]
    })
    df.write_parquet("raw/inventory.parquet")
    print("raw/inventory.parquet created.")

if __name__ == "__main__":
    create_inventory_data()
```

```python
python create_raw_data.py
```

**`docs/specs.pdf`:** Vamos gerar um PDF com texto fictício. Instale `fpdf2` (`poetry add fpdf2`) e use o script `create_spec_pdf.py` na raiz.

```python
# create_spec_pdf.py
from fpdf import FPDF
import os

PDF_TEXT = [
    (1, """
    Alpha-X Pro Router - Technical Specifications

    1. Interfaces
    - WAN: 1x 2.5 Gigabit Ethernet (RJ-45)
    - LAN: 4x 1 Gigabit Ethernet (RJ-45)
    - USB: 1x USB 3.0 for external storage or modem.
    """),
    (2, """
    2. Power Consumption
    - Idle: 5W
    - Maximum Load: 25W
    - Power Adapter: 12V, 2.5A DC

    3. Operational Environment
    - Operating temperature: 0°C to 40°C
    - Storage temperature: -20°C to 60°C
    - Humidity: 10% to 90% non-condensing
    """),
    (3, """
    4. Firmware Update
    - Method: Automatic via cloud or manual upload via Web UI.
    - Procedure: Download firmware from official site, navigate to the 'System' -> 'Firmware Update' section of the Web UI, and select the file. The device will restart after the update.
    - A USB drive can also be used for recovery.
    """),
    (4, """
    5. Certifications
    - FCC, CE, RoHS
    - Wi-Fi Alliance Certified
    """)
]

def create_pdf():
    if not os.path.exists("docs"):
        os.makedirs("docs")

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    for page_num, text in PDF_TEXT:
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 10, text)
        pdf.set_font("Arial", 'I', 8)
        pdf.cell(0, 10, f"Page {page_num}", 0, 0, 'C')

    pdf.output("docs/specs.pdf")
    print("docs/specs.pdf created.")

if __name__ == "__main__":
    create_pdf()
```

```python
python create_spec_pdf.py
```

ETL - Pipeline de Limpeza e Normalização

**`schemas/contracts.py`:** Defina os schemas de validação com Pandera.

```python
# schemas/contracts.py
import pandera.polars as pa
from pandera import Field

# Schema para a tabela de produtos limpa e validada
dim_product_schema = pa.DataFrameSchema({
    "product_id": pa.Column(str, checks=pa.Check.not_null(), unique=True),
    "sku": pa.Column(str, checks=pa.Check.not_null(), unique=True),
    "model": pa.Column(str, nullable=True),
    "category": pa.Column(str),
    "weight_g": pa.Column(int, checks=pa.Check.gt(0), nullable=True),
    "length_mm": pa.Column(int, checks=pa.Check.gt(0)),
    "width_mm": pa.Column(int, checks=pa.Check.gt(0)),
    "height_mm": pa.Column(int, checks=pa.Check.gt(0)),
    "vendor_code": pa.Column(str),
    "launch_date": pa.Column(pa.Date, nullable=True),
    "msrp_usd": pa.Column(float, checks=pa.Check.ge(0.0), coerce=True)
})

# Schema para a tabela de vendors consolidada
dim_vendor_schema = pa.DataFrameSchema({
    "vendor_code": pa.Column(str, checks=pa.Check.not_null(), unique=True),
    "vendor_name": pa.Column(str),
    "country": pa.Column(str),
    "support_email": pa.Column(str, checks=pa.Check.str_matches(r".+@.+\..+"))
})

# Schema para a tabela de fatos de inventário
fact_inventory_schema = pa.DataFrameSchema({
    "product_id": pa.Column(str, checks=pa.Check.not_null()),
    "warehouse": pa.Column(str),
    "on_hand": pa.Column(int, checks=pa.Check.ge(0)),
    "min_stock": pa.Column(int, checks=pa.Check.ge(0)),
    "last_counted_at": pa.Column(pa.DateTime)
})
```

Pipeline principal:

```python
# src/etl/run.py
import polars as pl
import hashlib
import os
import sys
from datetime import datetime

from loguru import logger
from pandera.polars import DataFrameSchema
from pandera.errors import SchemaErrors

from schemas.contracts import (
    dim_product_schema,
    dim_vendor_schema,
    fact_inventory_schema,
)

logger.remove()
logger.add(sys.stderr, level="INFO", format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>")

RAW_DATA_PATH = "raw"
SILVER_DATA_PATH = "silver"
QUARANTINE_PATH = os.path.join(SILVER_DATA_PATH, "_quarantine")
FINAL_DATA_PATH = "data"

def validate_and_quarantine(df: pl.DataFrame, schema: DataFrameSchema, table_name: str) -> pl.DataFrame:
    logger.info(f"Validating table '{table_name}'...")
    try:
        valid_df = schema.validate(df, lazy=True)
        logger.success(f"Validation successful for '{table_name}'. All {len(valid_df)} records are valid.")
        return valid_df
    except SchemaErrors as err:
        logger.warning(f"Validation failed for '{table_name}'. Found {len(err.failure_cases)} invalid records.")
        quarantine_df = err.failure_cases
        os.makedirs(QUARANTINE_PATH, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        quarantine_file = os.path.join(QUARANTINE_PATH, f"{table_name}_invalid_{timestamp}.csv")
        quarantine_df.write_csv(quarantine_file)
        logger.info(f"Quarantined records saved to {quarantine_file}")
        invalid_indexes = quarantine_df["index"]
        df_with_index = df.with_row_index(name="original_index")
        valid_df = df_with_index.filter(~pl.col("original_index").is_in(invalid_indexes)).drop("original_index")
        logger.info(f"Kept {len(valid_df)} valid records for '{table_name}'.")
        return valid_df

def process_products() -> pl.DataFrame:
    logger.info("Processing products...")
    df = pl.read_csv(f"{RAW_DATA_PATH}/products.csv", schema_overrides={"product_id": pl.String})

    df = df.with_columns(
        pl.when(pl.col("product_id").is_null())
        .then(pl.col("sku").map_elements(lambda s: hashlib.sha1(s.encode()).hexdigest()[:8], return_dtype=pl.String))
        .otherwise(pl.col("product_id"))
        .alias("product_id"),
        
     
        pl.col("msrp_usd")
        .cast(pl.String) # 1. Garante que é uma string
        .str.replace_all(",", ".") # 2. Limpa a string
        .cast(pl.Float64, strict=False) # 3. Converte para float
        .alias("msrp_usd"),
        
        pl.col("launch_date").str.to_date(format="%Y-%m-%d", strict=False)
    )

    dims = df["dimensions_mm"].str.split("x").map_elements(
        lambda s: [int(d) if d else None for d in s] + [None] * (3 - len(s)),
        return_dtype=pl.List(pl.Int64)
    )
    df = df.with_columns(
        length_mm=dims.list.get(0),
        width_mm=dims.list.get(1),
        height_mm=dims.list.get(2)
    ).drop("dimensions_mm")
    
    df = df.rename({"weight_grams": "weight_g"})

    validated_df = validate_and_quarantine(df, dim_product_schema, "dim_product")
    
    output_path = f"{FINAL_DATA_PATH}/dim_product"
    os.makedirs(output_path, exist_ok=True)
    validated_df.write_parquet(f"{output_path}/dim_product.parquet")
    logger.success(f"dim_product saved to {output_path}")
    return validated_df

def process_vendors():
    logger.info("Processing vendors...")
    df = pl.read_ndjson(f"{RAW_DATA_PATH}/vendors.jsonl")
    df = df.group_by("vendor_code").agg(
        pl.last("name").alias("vendor_name"),
        pl.last("country"),
        pl.last("support_email")
    )
    validated_df = validate_and_quarantine(df, dim_vendor_schema, "dim_vendor")
    output_path = f"{FINAL_DATA_PATH}/dim_vendor"
    os.makedirs(output_path, exist_ok=True)
    validated_df.write_parquet(f"{output_path}/dim_vendor.parquet")
    logger.success(f"dim_vendor saved to {output_path}")

def process_inventory(valid_product_ids: pl.Series):
    logger.info("Processing inventory...")
    df = pl.read_parquet(f"{RAW_DATA_PATH}/inventory.parquet")
    df = df.filter(pl.col("product_id").cast(pl.String).is_in(valid_product_ids))
    df = df.with_columns(pl.col("product_id").cast(pl.String))
    validated_df = validate_and_quarantine(df, fact_inventory_schema, "fact_inventory")
    output_path = f"{FINAL_DATA_PATH}/fact_inventory"
    os.makedirs(output_path, exist_ok=True)
    validated_df.write_parquet(f"{output_path}/fact_inventory.parquet")
    logger.success(f"fact_inventory saved to {output_path}")

def main():
    logger.info("Starting ETL process...")
    os.makedirs(FINAL_DATA_PATH, exist_ok=True)
    valid_products = process_products()
    valid_product_ids = valid_products.get_column("product_id")
    process_vendors()
    process_inventory(valid_product_ids)
    logger.success("ETL process finished successfully.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception("An unexpected error occurred during the ETL process.")
        sys.exit(1)
```

Vá até a pasta raiz onde se encontra o makefile e execute o seguinte comando:

```python
make run-etl
```

![image.png](Desafio%20EWork%2028f2e2f65d7e813689cdf9e52ba109d0/image%203.png)

---

Mini-RAG - Ingestão e Indexação

**`src/rag/ingest.py`:** Script para ler o PDF, chunkear, criar embeddings e salvar o índice FAISS.

```python
# src/rag/ingest.py
import os
import pickle
import faiss
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

PDF_PATH = "docs/specs.pdf"
DB_PATH = "db"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def ingest_data():
    """Lê o PDF, cria chunks, embeddings e salva um índice FAISS."""
    print("Starting data ingestion...")

    # 1. Ler o PDF
    reader = PdfReader(PDF_PATH)
    documents = []
    for i, page in enumerate(reader.pages):
        documents.append({
            "text": page.extract_text(),
            "page": i + 1
        })
    print(f"Loaded {len(documents)} pages from PDF.")

    # 2. Fazer o chunking
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        length_function=len
    )

    chunks = []
    for doc in documents:
        split_texts = text_splitter.split_text(doc["text"])
        for text in split_texts:
            chunks.append({
                "snippet": text,
                "page": doc["page"]
            })
    print(f"Created {len(chunks)} text chunks.")

    # 3. Gerar embeddings
    print(f"Loading embedding model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode([chunk["snippet"] for chunk in chunks], show_progress_bar=True)

    # 4. Criar e salvar o índice FAISS
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    os.makedirs(DB_PATH, exist_ok=True)
    faiss.write_index(index, f"{DB_PATH}/vector_index.faiss")
    print(f"FAISS index saved to {DB_PATH}/vector_index.faiss")

    # 5. Salvar os chunks para recuperação posterior
    with open(f"{DB_PATH}/chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)
    print(f"Chunks saved to {DB_PATH}/chunks.pkl")

    print("Ingestion complete.")

if __name__ == "__main__":
    ingest_data()
```

Vamos instalar alguns pacotes

```python

poetry add langchain-text-splitters

# Caso tenha algum problema com a lib do numpy
poetry add "numpy<2.0"

```

Executar o RAG

```python
make run-rag-ingest
```

![image.png](Desafio%20EWork%2028f2e2f65d7e813689cdf9e52ba109d0/image%204.png)

**Mini-RAG - API com FastAPI**

**`src/rag/api.py`:** Código da API que usa o índice para responder perguntas.

```python
# src/rag/api.py
import os
import pickle
import faiss
import requests
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from contextlib import asynccontextmanager

# --- Modelos Pydantic ---
class AskRequest(BaseModel):
    question: str
    top_k: int = 3

class Source(BaseModel):
    page: int
    snippet: str

class AskResponse(BaseModel):
    answer: str
    sources: list[Source]

# --- Variáveis Globais e Configuração ---
DB_PATH = "db"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
OLLAMA_API_URL = "http://localhost:11434/api/generate"

ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Carregar recursos na inicialização
    print("Loading RAG model and data...")
    ml_models["embedding_model"] = SentenceTransformer(MODEL_NAME)
    ml_models["index"] = faiss.read_index(f"{DB_PATH}/vector_index.faiss")
    with open(f"{DB_PATH}/chunks.pkl", "rb") as f:
        ml_models["chunks"] = pickle.load(f)
    print("Resources loaded successfully.")
    yield
    # Limpar recursos no desligamento
    ml_models.clear()

app = FastAPI(lifespan=lifespan)

def retrieve_context(question: str, top_k: int) -> list[dict]:
    model = ml_models["embedding_model"]
    index = ml_models["index"]
    chunks = ml_models["chunks"]
    question_embedding = model.encode([question])
    distances, indices = index.search(question_embedding, top_k)
    relevant_chunks = [chunks[i] for i in indices[0]]
    return relevant_chunks

def generate_answer(question: str, context: list[dict]) -> str:
    context_str = "\n---\n".join([f"Fonte (Página {c['page']}):\n{c['snippet']}" for c in context])

    prompt = f"""
    Você é um assistente técnico especialista. Baseado ESTREITAMENTE no contexto fornecido abaixo, responda a pergunta do usuário.
    Se a resposta não estiver no contexto, responda: "Não encontrei informações sobre isso nas especificações técnicas."
    Cite sempre a página da fonte.

    **Contexto:**
    {context_str}

    **Pergunta:**
    {question}

    **Resposta:**
    """
    try:
        response = requests.post(
            OLLAMA_API_URL,
            json={"model": "llama3.1", "prompt": prompt, "stream": False},
            timeout=120
        )
        response.raise_for_status()
        return response.json()["response"]
    except requests.RequestException as e:
        print(f"Error calling Ollama API: {e}")
        raise HTTPException(status_code=500, detail="Failed to communicate with the language model.")

@app.post("/ask", response_model=AskResponse)
def ask_question(request: AskRequest):
    if not all(k in ml_models for k in ["embedding_model", "index", "chunks"]):
        raise HTTPException(status_code=503, detail="Model and data not loaded yet. Please wait.")

    context_chunks = retrieve_context(request.question, request.top_k)

    if not context_chunks:
        return AskResponse(
            answer="Não foi possível encontrar trechos relevantes no documento para responder a sua pergunta.",
            sources=[]
        )

    answer = generate_answer(request.question, context_chunks)
    sources = [Source(page=chunk["page"], snippet=chunk["snippet"]) for chunk in context_chunks]
    return AskResponse(answer=answer, sources=sources)

@app.get("/health")
def health_check():
    return {"status": "ok"}
```

Vamos executar essa API

```python
make run-rag-api
```

![image.png](Desafio%20EWork%2028f2e2f65d7e813689cdf9e52ba109d0/image%205.png)

OBS 🚨: O Teste do /POST a seguir foi testado em um terminal Linux

```python
curl -X POST "http://localhost:8003/ask" \
-H "Content-Type: application/json" \
-d '{"question": "Qual é a faixa de temperatura operacional?"}'
```

![image.png](Desafio%20EWork%2028f2e2f65d7e813689cdf9e52ba109d0/image%206.png)

---

### **Testes Automatizados**

Teste ETL

```python
# tests/test_etl.py
import os
import shutil
import polars as pl
from src.etl.run import main as run_etl

TEST_RAW_PATH = "tests/temp_raw"
TEST_SILVER_PATH = "silver"
TEST_DATA_PATH = "data"

def setup_function():
    os.makedirs(TEST_RAW_PATH, exist_ok=True)
    if os.path.exists(TEST_SILVER_PATH): shutil.rmtree(TEST_SILVER_PATH)
    if os.path.exists(TEST_DATA_PATH): shutil.rmtree(TEST_DATA_PATH)

def teardown_function():
    shutil.rmtree(TEST_RAW_PATH)
    if os.path.exists(TEST_SILVER_PATH): shutil.rmtree(TEST_SILVER_PATH)
    if os.path.exists(TEST_DATA_PATH): shutil.rmtree(TEST_DATA_PATH)

def test_quarantine_invalid_product_date():
    """
    Testa se um produto com dados inválidos (peso negativo) é enviado para a quarentena.
    """
    # CORREÇÃO DEFINITIVA: Em vez de uma data inválida (que é convertida para null),
    # usamos um peso negativo, que vai falhar na regra de validação `checks=pa.Check.gt(0)`.
    bad_product_data = """product_id,sku,model,category,weight_grams,dimensions_mm,vendor_code,launch_date,msrp_usd
1002,AB-002,Alpha-X Pro,Router,-50,"220x120x45",V-77,2023-12-01,159.90
"""
    with open(f"{TEST_RAW_PATH}/products.csv", "w") as f:
        f.write(bad_product_data)

    dummy_vendor_df = pl.DataFrame({
        "vendor_code": ["V-77"], "name": ["Test Vectortron"],
        "country": ["DE"], "support_email": ["test@vectortron.com"]
    })
    dummy_vendor_df.write_ndjson(f"{TEST_RAW_PATH}/vendors.jsonl")
    
    pl.DataFrame({"product_id": ["1002"]}).write_parquet(f"{TEST_RAW_PATH}/inventory.parquet")

    import src.etl.run
    src.etl.run.RAW_DATA_PATH = TEST_RAW_PATH
    src.etl.run.SILVER_DATA_PATH = TEST_SILVER_PATH
    src.etl.run.FINAL_DATA_PATH = TEST_DATA_PATH

    run_etl()

    quarantine_dir = os.path.join(TEST_SILVER_PATH, "_quarantine")
    assert os.path.exists(quarantine_dir), "A pasta de quarentena não foi criada."
    
    files = os.listdir(quarantine_dir)
    assert any("dim_product_invalid" in f for f in files), "Arquivo de quarentena para produtos não encontrado."
```

Teste RAG

```python
# tests/test_rag.py
from fastapi.testclient import TestClient
from src.rag.api import app

def test_health_check():
    # Usar o 'with' garante que o lifespan (startup/shutdown) seja executado
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

def test_ask_endpoint_smoke_test():
    """Testa se o endpoint /ask responde corretamente a uma pergunta válida."""
    # Usar o 'with' garante que o lifespan (startup/shutdown) seja executado
    with TestClient(app) as client:
        # Este teste depende que o `make run-rag-ingest` tenha sido executado
        response = client.post(
            "/ask",
            json={"question": "Qual a temperatura de operação?"}
        )
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "sources" in data
        assert isinstance(data["answer"], str)
        assert isinstance(data["sources"], list)
        assert len(data["answer"]) > 0
```

Agora execute os testes:

```python
make test 
```

![image.png](Desafio%20EWork%2028f2e2f65d7e813689cdf9e52ba109d0/image%207.png)

Teste realizado com sucesso.

---

### Bonus:

Criei um Docker File para API RAG

```docker

FROM python:3.12-slim

WORKDIR /app

RUN pip install poetry
RUN poetry config virtualenvs.create false

COPY pyproject.toml poetry.lock ./

RUN poetry install --no-root --without dev

COPY . .

# Passo 6: Comando Padrão
CMD ["poetry", "run", "uvicorn", "src.rag.api:app", "--host", "0.0.0.0", "--port", "8003"]
```

```docker
docker build -t erickwolf/ework-desafio-api:latest .
docker push erickwolf/ework-desafio-api:latest
```

![image.png](Desafio%20EWork%2028f2e2f65d7e813689cdf9e52ba109d0/image%208.png)

![image.png](Desafio%20EWork%2028f2e2f65d7e813689cdf9e52ba109d0/image%209.png)
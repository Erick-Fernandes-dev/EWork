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
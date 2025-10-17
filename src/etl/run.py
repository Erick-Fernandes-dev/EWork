# src/etl/run.py (VERSÃO FINAL E ROBUSTA)
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
        
        # LÓGICA FINAL E ROBUSTA: Cast -> Clean -> Cast
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
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
    df.write_parquet("./inventory.parquet")
    print("./inventory.parquet created.")

if __name__ == "__main__":
    create_inventory_data()
# schemas/contracts.py (CORRIGIDO)
import pandera.polars as pa
import polars as pl  # <-- Adicione esta importação
from pandera import Field

# Schema para a tabela de produtos limpa e validada
dim_product_schema = pa.DataFrameSchema({
    "product_id": pa.Column(str, nullable=False, unique=True),
    "sku": pa.Column(str, nullable=False, unique=True),
    "model": pa.Column(str, nullable=True),
    "category": pa.Column(str),
    "weight_g": pa.Column(int, checks=pa.Check.gt(0), nullable=True),
    "length_mm": pa.Column(int, checks=pa.Check.gt(0)),
    "width_mm": pa.Column(int, checks=pa.Check.gt(0)),
    "height_mm": pa.Column(int, checks=pa.Check.gt(0)),
    "vendor_code": pa.Column(str),
    # CORREÇÃO: Trocado pa.Date por pl.Date
    "launch_date": pa.Column(pl.Date, nullable=True),
    "msrp_usd": pa.Column(float, checks=pa.Check.ge(0.0), coerce=True)
})

# Schema para a tabela de vendors consolidada
dim_vendor_schema = pa.DataFrameSchema({
    "vendor_code": pa.Column(str, nullable=False, unique=True),
    "vendor_name": pa.Column(str),
    "country": pa.Column(str),
    "support_email": pa.Column(str, checks=pa.Check.str_matches(r".+@.+\..+"))
})

# Schema para a tabela de fatos de inventário
fact_inventory_schema = pa.DataFrameSchema({
    "product_id": pa.Column(str, nullable=False),
    "warehouse": pa.Column(str),
    "on_hand": pa.Column(int, checks=pa.Check.ge(0)),
    "min_stock": pa.Column(int, checks=pa.Check.ge(0)),
    # CORREÇÃO: Trocado pa.DateTime por pl.Datetime
    "last_counted_at": pa.Column(pl.Datetime)
})
import pandas as pd
import os
import json
import re
from hs_lookup import get_hs_lookup, match_hs_description

# =========================
# CONFIG
# =========================

DATA_URL = "https://huggingface.co/datasets/WilgnerCH/canada-trade-data/resolve/main/canada_trade_full.parquet"

OUTPUT_DIR = "data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_CSV_DIR = "data_csv"
os.makedirs(OUTPUT_CSV_DIR, exist_ok=True)


# =========================
# LOAD DATA
# =========================

def load_data():
    print("📥 Loading dataset...")
    df = pd.read_parquet(DATA_URL)
    print(f"✅ Dataset loaded: {len(df)} rows")
    return df


# =========================
# CLEAN (REMOVE DUPLICATION)
# =========================

def clean_data(df):
    print("🧹 Cleaning data (removing duplicates)...")

    df_clean = (
        df.groupby(["date", "trade_type", "Country", "HS"])["Value"]
        .sum()
        .reset_index()
    )

    print(f"✅ Clean dataset: {len(df_clean)} rows")
    return df_clean


# =========================
# CLEAN FOR PROVINCES (keeps Province column)
# =========================

def clean_data_province(df):
    print("🧹 Cleaning data for provinces (removing duplicates)...")

    df_clean = (
        df.groupby(["date", "trade_type", "Province", "HS"])["Value"]
        .sum()
        .reset_index()
    )

    print(f"✅ Clean province dataset: {len(df_clean)} rows")
    return df_clean


# =========================
# MONTHLY SUMMARY
# =========================

def monthly_summary(df):

    monthly = (
        df.groupby(["date", "trade_type"])["Value"]
        .sum()
        .reset_index()
    )

    print("📊 Monthly summary created")
    return monthly


# =========================
# COUNTRY SUMMARY
# =========================

def country_summary(df):

    country = (
        df.groupby(["Country", "trade_type"])["Value"]
        .sum()
        .reset_index()
    )

    print("🌍 Country summary created")
    return country


# =========================
# PRODUCTS SUMMARY
# =========================

def product_summary(df):

    products = (
        df.groupby(["HS", "trade_type"])["Value"]
        .sum()
        .reset_index()
    )

    print("📦 Product summary created")
    return products


# =========================
# PROVINCE SUMMARY
# =========================

def province_summary(df_prov):

    province = (
        df_prov.groupby(["Province", "trade_type"])["Value"]
        .sum()
        .reset_index()
    )

    print("🍁 Province summary created")
    return province


# =========================
# PROVINCE MONTHLY SUMMARY
# =========================

def province_monthly_summary(df_prov):

    province_monthly = (
        df_prov.groupby(["date", "Province", "trade_type"])["Value"]
        .sum()
        .reset_index()
    )

    print("🍁 Province monthly summary created")
    return province_monthly


# =========================
# PROVINCE x HS2 SUMMARY
# =========================

def province_hs2_summary(df_prov):

    df_prov = df_prov.copy()

    # Extract HS2 chapter (first 2 digits of "XXXX.XX.XX")
    df_prov["HS2"] = df_prov["HS"].str.replace(".", "", regex=False).str[:2]

    province_hs2 = (
        df_prov.groupby(["Province", "HS2", "trade_type"])["Value"]
        .sum()
        .reset_index()
    )

    print("🍁 Province x HS2 summary created")
    return province_hs2


# =========================
# 🧼 CLEAN PRODUCT NAME
# =========================

def clean_product_name(name):
    if not name:
        return name

    # Remove HS code no início (ex: 2709.00, 01.01, 0101 etc.)
    cleaned = re.sub(r'^\d+(\.\d+)*\s*', '', name)

    return cleaned.strip()


# =========================
# SAVE OUTPUTS
# =========================

def save_outputs(monthly, country, products, province, province_monthly, province_hs2, hs_lookup):

    # -------------------------
    # MONTHLY JSON
    # -------------------------
    monthly_pivot = (
        monthly.pivot(index="date", columns="trade_type", values="Value")
        .fillna(0)
        .reset_index()
    )

    monthly_json = [
        {
            "date": row["date"],
            "imports": float(row.get("Import", 0)),
            "exports": float(row.get("Export", 0))
        }
        for _, row in monthly_pivot.iterrows()
    ]

    # -------------------------
    # COUNTRIES JSON
    # -------------------------
    countries_pivot = (
        country.pivot(index="Country", columns="trade_type", values="Value")
        .fillna(0)
        .reset_index()
    )

    countries_json = [
        {
            "country": row["Country"],
            "imports": float(row.get("Import", 0)),
            "exports": float(row.get("Export", 0)),
            "total": float(row.get("Import", 0) + row.get("Export", 0))
        }
        for _, row in countries_pivot.iterrows()
    ]

    countries_json = sorted(countries_json, key=lambda x: x["total"], reverse=True)

    # -------------------------
    # PRODUCTS JSON (COM NOME LIMPO)
    # -------------------------
    products_pivot = (
        products.pivot(index="HS", columns="trade_type", values="Value")
        .fillna(0)
        .reset_index()
    )

    products_json = []

    for _, row in products_pivot.iterrows():

        hs_code = row["HS"]

        # Nome original
        raw_name = match_hs_description(hs_code, hs_lookup)

        # Nome limpo
        name = clean_product_name(raw_name)

        products_json.append({
            "hs": str(hs_code),
            "name": name,
            "imports": float(row.get("Import", 0)),
            "exports": float(row.get("Export", 0)),
            "total": float(row.get("Import", 0) + row.get("Export", 0))
        })

    products_json = sorted(products_json, key=lambda x: x["total"], reverse=True)

    # -------------------------
    # PROVINCES CSV
    # -------------------------
    province_pivot = (
        province.pivot(index="Province", columns="trade_type", values="Value")
        .fillna(0)
        .reset_index()
    )

    province_rows = []
    for _, row in province_pivot.iterrows():
        exports = float(row.get("Export", 0))
        imports = float(row.get("Import", 0))
        province_rows.append({
            "province": row["Province"],
            "exports": exports,
            "imports": imports,
            "total": exports + imports
        })

    province_df = pd.DataFrame(province_rows).sort_values("total", ascending=False)
    province_df.to_csv(f"{OUTPUT_CSV_DIR}/provinces.csv", index=False)
    print(f"🍁 provinces.csv written: {len(province_df)} rows")

    # -------------------------
    # PROVINCES MONTHLY CSV
    # -------------------------
    province_monthly_pivot = (
        province_monthly.pivot_table(
            index=["date", "Province"], columns="trade_type", values="Value"
        )
        .fillna(0)
        .reset_index()
    )

    province_monthly_rows = []
    for _, row in province_monthly_pivot.iterrows():
        exports = float(row.get("Export", 0))
        imports = float(row.get("Import", 0))
        province_monthly_rows.append({
            "date": row["date"],
            "province": row["Province"],
            "exports": exports,
            "imports": imports,
            "total": exports + imports
        })

    province_monthly_df = pd.DataFrame(province_monthly_rows).sort_values(["province", "date"])
    province_monthly_df.to_csv(f"{OUTPUT_CSV_DIR}/provinces_monthly.csv", index=False)
    print(f"🍁 provinces_monthly.csv written: {len(province_monthly_df)} rows")

    # -------------------------
    # PROVINCES x HS2 CSV (com nome limpo)
    # -------------------------
    province_hs2_pivot = (
        province_hs2.pivot_table(
            index=["Province", "HS2"], columns="trade_type", values="Value"
        )
        .fillna(0)
        .reset_index()
    )

    province_hs2_rows = []
    for _, row in province_hs2_pivot.iterrows():
        hs2_code = row["HS2"]

        # Reaproveita o HS lookup — usa "XX.00.00" para casar com a tabela de descrições
        raw_name = match_hs_description(f"{hs2_code}.00.00", hs_lookup)
        name = clean_product_name(raw_name)

        exports = float(row.get("Export", 0))
        imports = float(row.get("Import", 0))

        province_hs2_rows.append({
            "province": row["Province"],
            "hs2": hs2_code,
            "hs2_name": name,
            "exports": exports,
            "imports": imports,
            "total": exports + imports
        })

    province_hs2_df = pd.DataFrame(province_hs2_rows).sort_values(
        ["province", "total"], ascending=[True, False]
    )
    province_hs2_df.to_csv(f"{OUTPUT_CSV_DIR}/provinces_hs2.csv", index=False)
    print(f"🍁 provinces_hs2.csv written: {len(province_hs2_df)} rows")

    # -------------------------
    # SAVE JSON FILES
    # -------------------------
    with open(f"{OUTPUT_DIR}/monthly.json", "w") as f:
        json.dump(monthly_json, f)

    with open(f"{OUTPUT_DIR}/countries.json", "w") as f:
        json.dump(countries_json[:20], f)

    with open(f"{OUTPUT_DIR}/products.json", "w") as f:
        json.dump(products_json[:20], f)

    print("💾 JSON files saved in /data")
    print("💾 Province CSV files saved in /data_csv")


# =========================
# MAIN
# =========================

def main():

    df = load_data()

    # Clean dataset for general aggregations (no Province)
    df_clean = clean_data(df)

    # Clean dataset for province aggregations (keeps Province)
    df_prov_clean = clean_data_province(df)

    m = monthly_summary(df_clean)
    c = country_summary(df_clean)
    p = product_summary(df_clean)

    province = province_summary(df_prov_clean)
    province_monthly = province_monthly_summary(df_prov_clean)
    province_hs2 = province_hs2_summary(df_prov_clean)

    print("🔗 Loading HS lookup...")
    hs_lookup = get_hs_lookup()

    save_outputs(m, c, p, province, province_monthly, province_hs2, hs_lookup)

    print("🚀 Pipeline finished successfully!")


if __name__ == "__main__":
    main()

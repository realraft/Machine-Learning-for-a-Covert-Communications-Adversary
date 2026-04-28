import pandas as pd
import numpy as np

CSV_PATH = "C:\\Users\\oaraf\\Desktop\\Honors Thesis\\data\\data.csv"

df = pd.read_csv(CSV_PATH)

print("=== UNIQUENESS CHECK ===")
print(f"Total rows:        {len(df)}")
print(f"Unique rows:       {df.drop(columns='nonlinear').drop_duplicates().shape[0]}")
print(f"Exact duplicates:  {df.duplicated(subset=df.columns[:-1]).sum()}")

# How many unique values per feature?
print("\n=== UNIQUE VALUES PER FEATURE ===")
for col in df.columns[:-1]:
    print(f"  {col:25s}: {df[col].nunique()} unique values")

# Are duplicate feature vectors always the same class?
print("\n=== DUPLICATE ROWS — DO THEY SHARE A LABEL? ===")
dupes = df[df.duplicated(subset=df.columns[:-1], keep=False)]
print(f"Rows involved in duplicates: {len(dupes)}")
if len(dupes) > 0:
    conflict = dupes.groupby(list(df.columns[:-1]))['nonlinear'].nunique()
    print(f"Duplicate groups with SAME label:      {(conflict == 1).sum()}")
    print(f"Duplicate groups with CONFLICT label:  {(conflict > 1).sum()}")

# Check how many truly independent samples exist
print("\n=== NEAR-DUPLICATE CHECK (rounded to 3 decimal places) ===")
df_rounded = df.drop(columns='nonlinear').round(3)
print(f"Unique rows after rounding to 3dp: {df_rounded.drop_duplicates().shape[0]}")
df_rounded2 = df.drop(columns='nonlinear').round(2)
print(f"Unique rows after rounding to 2dp: {df_rounded2.drop_duplicates().shape[0]}")
df_rounded1 = df.drop(columns='nonlinear').round(1)
print(f"Unique rows after rounding to 1dp: {df_rounded1.drop_duplicates().shape[0]}")
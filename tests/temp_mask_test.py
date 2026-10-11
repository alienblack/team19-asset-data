import pandas as pd

df = pd.read_csv(
    "data/synthetic/overhead_conductors/"
    "masked/MF01/masked_MF01.csv"
)

columns = [
    "material",
    "installation_year",
    "voltage_category",
    "corrosion_zone",
    "covering",
    "length_km",
    "location_group",
    "distance_to_sea_km",
]

print(df[columns].isna().sum())

print("\nRows with all 8 model fields missing:")
print(df[columns].isna().all(axis=1).sum())

print("\nMissing percentage:")
print(df[columns].isna().mean() * 100)
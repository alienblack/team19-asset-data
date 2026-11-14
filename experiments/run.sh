@ -0,0 +1,66 @@
PYTHONPATH=src python \
src/team19_asset_data/overhead_conductors/run_masking.py \
  --rules MF01 \
  --input data/synthetic/overhead_conductors/strength_100/material_model_ready_data.csv \
  --config config/overhead_conductor_masking_rules.yaml \
  --output-dir data/synthetic/overhead_conductors/strength_100/masked/MF01

PYTHONPATH=src python -m \
experiments.run_imputation_experiment \
  --input data/synthetic/overhead_conductors/strength_100/masked/MF01/masked_MF01.csv \
  --clean-source data/synthetic/overhead_conductors/strength_100/material_model_ready_data.csv \
  --config config/imputation/overhead_conductors.yaml \
  --mask-metadata data/synthetic/overhead_conductors/strength_100/masked/MF01/metadata_MF01.json \
  --output-dir outputs/imputation_experiments/overhead_conductors/strength_100/noise_0/MF01 \
  --relationship-strength 100 \
  --noise-level 0 \
  --mask-rule MF01



PYTHONPATH=src python -m \
team19_asset_data.overhead_conductors.run_noise \
  --input data/synthetic/overhead_conductors/strength_100/material_model_ready_data.csv \
  --config config/overhead_conductor_noise_rules.yaml \
  --output-dir data/synthetic/overhead_conductors/strength_100/noise \
  --scenarios N01_20 N01_30 N01_50


for scenario in N01_20 N01_30 N01_50; do
  echo "Applying MF01 masking to ${scenario}..."

  PYTHONPATH=src python \
    src/team19_asset_data/overhead_conductors/run_masking.py \
    --rules MF01 \
    --input "data/synthetic/overhead_conductors/strength_100/noise/${scenario}/observed.csv" \
    --config config/overhead_conductor_masking_rules.yaml \
    --output-dir "data/synthetic/overhead_conductors/strength_100/noise/${scenario}/masked/MF01"
done



for scenario in N01_20 N01_30 N01_50; do
  level=${scenario#N01_}

  echo "Running imputation: ${level}% noise"

  PYTHONPATH=src python -m \
    experiments.run_imputation_experiment \
    --input "data/synthetic/overhead_conductors/strength_100/noise/${scenario}/masked/MF01/masked_MF01.csv" \
    --clean-source data/synthetic/overhead_conductors/strength_100/material_model_ready_data.csv \
    --config config/imputation/overhead_conductors.yaml \
    --mask-metadata "data/synthetic/overhead_conductors/strength_100/noise/${scenario}/masked/MF01/metadata_MF01.json" \
    --output-dir "outputs/imputation_experiments/overhead_conductors/strength_100/noise_${level}/MF01" \
    --relationship-strength 100 \
    --noise-level "0.${level}" \
    --mask-rule MF01
done



PYTHONPATH=src python experiments/run_predictive_imputation_experiment.py \
  --config config/imputation/overhead_conductors_prototype_v2.yaml \
  --input-csv data/synthetic/overhead_conductors/strength_100/material_model_ready_data.csv \
  --output-dir outputs/predictive_smoke \
  --sample-rows 350 \
  --n-estimators 15


  PYTHONPATH=src python -m experiments.run_predictive_imputation_experiment \
  --input-csv data/synthetic/overhead_conductors/strength_100/masked/MF01/masked_MF01.csv \
  --config config/imputation/overhead_conductors_prototype_v2.yaml \
  --output-dir outputs/predictive_missforest


  PYTHONPATH=src python -m experiments.run_predictive_imputation_experiment \
  --input-csv data/synthetic/overhead_conductors/strength_100/masked/MF01/masked_MF01.csv \
  --config config/imputation/overhead_conductors_tuning_v2.yaml \
  --output-dir outputs/predictive_missforest_tuned


  PYTHONPATH=src python -m experiments.run_predictive_imputation_experiment \
  --input-csv data/synthetic/overhead_conductors/strength_100/masked/MF01/masked_MF01.csv \
  --config config/imputation/overhead_conductors_all_attributes_v2.yaml \
  --output-dir outputs/predictive_missforest_all_attributes


  PYTHONPATH=src python -m experiments.demo_01_inspect \
  --clean-csv data/synthetic/overhead_conductors/strength_100/material_model_ready_data.csv \
  --input-csv data/synthetic/overhead_conductors/strength_100/masked/MF01/masked_MF01.csv \
  --config config/imputation/overhead_conductors_all_attributes_v2.yaml


  PYTHONPATH=src python -m experiments.demo_02_fit_transform \
  --input-csv data/synthetic/overhead_conductors/strength_100/masked/MF01/masked_MF01.csv \
  --config config/imputation/overhead_conductors_all_attributes_v2.yaml \
  --output-dir outputs/demo_predictive_missforest_e3


  PYTHONPATH=src python -m experiments.demo_03_transform_only \
  --input-csv data/synthetic/overhead_conductors/strength_100/masked/MF01/masked_MF01.csv \
  --model-path outputs/demo_predictive_missforest_e3/model_e3.joblib \
  --output-dir outputs/demo_predictive_missforest_e3 \
  --compare-to outputs/demo_predictive_missforest_e3/fit_transform_imputed.csv



python - <<'PY'
import pandas as pd
from sklearn.metrics import classification_report

original = pd.read_csv(
    "data/synthetic/overhead_conductors/strength_100/masked/MF01/masked_MF01.csv",
    dtype={"segment_id": str},
)

mask = pd.read_csv(
    "outputs/predictive_missforest/test_hidden_mask.csv",
    dtype={"segment_id": str},
)

predicted = pd.read_csv(
    "outputs/predictive_missforest/predictive_imputed_test.csv",
    dtype={"segment_id": str},
)

# Get IDs of records masked specifically for evaluation
ids = mask.loc[mask["material"], "segment_id"]

actual = original.set_index("segment_id").loc[ids, "material"]
prediction = predicted.set_index("segment_id").loc[ids, "material"]

print("Material confusion matrix:")
print(
    pd.crosstab(
        actual,
        prediction,
        rownames=["Actual"],
        colnames=["Predicted"],
    ).to_string()
)

print("\nClassification report:")
print(
    classification_report(
        actual,
        prediction,
        labels=["Steel", "Copper", "Aluminium"],
        zero_division=0,
    )
)
PY


PYTHONPATH=src python -m experiments.run_predictive_imputation_experiment \
  --input-csv data/synthetic/overhead_conductors/strength_100/masked/MF01/masked_MF01.csv \
  --config config/imputation/overhead_conductors_tuning_v2.yaml \
  --output-dir outputs/predictive_missforest_tuned

python - <<'PY'
import pandas as pd

untuned = pd.read_csv(
    "outputs/predictive_missforest/metrics_test.csv"
)

tuned = pd.read_csv(
    "outputs/predictive_missforest_tuned/metrics_test.csv"
)

for variable, metric in [
    ("installation_year", "mae"),
    ("material", "macro_f1"),
]:
    print(f"\n{variable} ({metric})")

    for name, df in [("Untuned", untuned), ("Tuned", tuned)]:
        row = df[
            (df["method"] == "predictive_missforest")
            & (df["variable"] == variable)
        ]
        print(f"{name}: {row[metric].iloc[0]:.4f}")
PY




====================Data Generation for Team =========================================================================


PYTHONPATH=src python -m team19_asset_data.overhead_conductors.run_masking \
  --rules MF01 \
  --input data/synthetic/overhead_conductors/strength_100/material_model_ready_data.csv \
  --config config/overhead_conductor_masking_rules.yaml \
  --output-dir outputs/team19_shared/masking_runs/noise_0

#Verify the regenerated dataset

python - <<'PY'
import pandas as pd

path = "outputs/team19_shared/masking_runs/noise_0/masked_MF01.csv"

df = pd.read_csv(path)

columns = [
    "installation_year",
    "distance_to_sea_km",
    "length_km",
    "voltage_category",
    "location_group",
    "corrosion_zone",
    "material",
    "covering",
]

print("Total records:", len(df))
print("\nMissing values per attribute:")
print(df[columns].isna().sum())

assert len(df) == 10000
assert df[columns].isna().sum().eq(2000).all()

print("\n0% noise + 20% missingness: PASSED")
PY


#generate only the 10% scenario first
PYTHONPATH=src python -m team19_asset_data.overhead_conductors.run_noise \
  --input data/synthetic/overhead_conductors/strength_100/material_model_ready_data.csv \
  --config config/overhead_conductor_noise_rules.yaml \
  --output-dir outputs/team19_shared/noisy_sources \
  --scenarios N06_mixed_10

  #Verify the regenerated dataset
  python - <<'PY'
import pandas as pd

root = "data/synthetic/overhead_conductors/strength_100"
noise = "outputs/team19_shared/noisy_sources/N06_mixed_10"

clean = pd.read_csv(f"{root}/material_model_ready_data.csv")
noisy = pd.read_csv(f"{noise}/observed.csv")

for column in ["installation_year", "material"]:
    changed = (clean[column] != noisy[column]).sum()
    print(f"{column}: {changed} changed values")

assert (
    noisy["asset_age"]
    == 2014 - noisy["installation_year"]
).all()

print("Year/asset-age consistency: PASSED")
PY

#Apply MF01 to the 10% noise dataset
PYTHONPATH=src python -m team19_asset_data.overhead_conductors.run_masking \
  --rules MF01 \
  --input outputs/team19_shared/noisy_sources/N06_mixed_10/observed.csv \
  --config config/overhead_conductor_masking_rules.yaml \
  --output-dir outputs/team19_shared/masking_runs/noise_10
#Verify the regenerated dataset
python - <<'PY'
import pandas as pd

base = "outputs/team19_shared/masking_runs"

reference = pd.read_csv(f"{base}/noise_0/masked_MF01.csv")
candidate = pd.read_csv(f"{base}/noise_10/masked_MF01.csv")

columns = [
    "installation_year",
    "distance_to_sea_km",
    "length_km",
    "voltage_category",
    "location_group",
    "corrosion_zone",
    "material",
    "covering",
]

assert len(reference) == len(candidate) == 10000
assert reference["segment_id"].equals(candidate["segment_id"])

print("=== 10% NOISE MASK VALIDATION ===")

for column in columns:
    reference_mask = reference[column].isna()
    candidate_mask = candidate[column].isna()

    count = int(candidate_mask.sum())
    identical = reference_mask.equals(candidate_mask)

    print(f"{column:<25} Missing: {count:<5} Same mask: {identical}")

    assert count == 2000
    assert identical, f"Mask mismatch: {column}"

assert reference["asset_age"].isna().equals(
    candidate["asset_age"].isna()
)

print("\n10% noise + 20% missingness: PASSED")
PY




#Generate 30% mixed noise
PYTHONPATH=src python -m team19_asset_data.overhead_conductors.run_noise \
  --input data/synthetic/overhead_conductors/strength_100/material_model_ready_data.csv \
  --config config/overhead_conductor_noise_rules.yaml \
  --output-dir outputs/team19_shared/noisy_sources \
  --scenarios N06_mixed_30

python - <<'PY'
import pandas as pd

clean = pd.read_csv(
    "data/synthetic/overhead_conductors/strength_100/material_model_ready_data.csv"
)

noisy = pd.read_csv(
    "outputs/team19_shared/noisy_sources/N06_mixed_30/observed.csv"
)

columns = [
    "installation_year",
    "distance_to_sea_km",
    "length_km",
    "voltage_category",
    "location_group",
    "corrosion_zone",
    "material",
    "covering",
]

print("\n=== 30% MIXED NOISE VALIDATION ===")

for column in columns:
    changed = int((clean[column] != noisy[column]).sum())
    print(f"{column:<25} {changed:>5} changed")

assert (
    noisy["asset_age"]
    == 2014 - noisy["installation_year"]
).all()

print("\nAsset-age consistency: PASSED")
PY

PYTHONPATH=src python -m team19_asset_data.overhead_conductors.run_masking \
  --rules MF01 \
  --input outputs/team19_shared/noisy_sources/N06_mixed_30/observed.csv \
  --config config/overhead_conductor_masking_rules.yaml \
  --output-dir outputs/team19_shared/masking_runs/noise_30

python - <<'PY'
import pandas as pd

base = "outputs/team19_shared/masking_runs"

reference = pd.read_csv(f"{base}/noise_0/masked_MF01.csv")
candidate = pd.read_csv(f"{base}/noise_30/masked_MF01.csv")

columns = [
    "installation_year",
    "distance_to_sea_km",
    "length_km",
    "voltage_category",
    "location_group",
    "corrosion_zone",
    "material",
    "covering",
]

assert len(reference) == len(candidate) == 10000
assert reference["segment_id"].equals(candidate["segment_id"])

print("=== 30% NOISE MASK VALIDATION ===")

for column in columns:
    reference_mask = reference[column].isna()
    candidate_mask = candidate[column].isna()

    missing = int(candidate_mask.sum())
    same_mask = reference_mask.equals(candidate_mask)

    print(
        f"{column:<25} "
        f"Missing: {missing:<5} "
        f"Same mask: {same_mask}"
    )

    assert missing == 2000
    assert same_mask, f"Mask mismatch: {column}"

assert reference["asset_age"].isna().equals(
    candidate["asset_age"].isna()
)

print("\n30% noise + 20% missingness: PASSED")
PY





#Generate 50% mixed noise

PYTHONPATH=src python -m team19_asset_data.overhead_conductors.run_noise \
  --input data/synthetic/overhead_conductors/strength_100/material_model_ready_data.csv \
  --config config/overhead_conductor_noise_rules.yaml \
  --output-dir outputs/team19_shared/noisy_sources \
  --scenarios N06_mixed_50

python - <<'PY'
import pandas as pd

clean = pd.read_csv(
    "data/synthetic/overhead_conductors/strength_100/material_model_ready_data.csv"
)

noisy = pd.read_csv(
    "outputs/team19_shared/noisy_sources/N06_mixed_50/observed.csv"
)

columns = [
    "installation_year",
    "distance_to_sea_km",
    "length_km",
    "voltage_category",
    "location_group",
    "corrosion_zone",
    "material",
    "covering",
]

print("\n=== 50% MIXED NOISE VALIDATION ===")

assert len(clean) == len(noisy) == 10000
assert clean["segment_id"].equals(noisy["segment_id"])

for column in columns:
    changed = int((clean[column] != noisy[column]).sum())
    print(f"{column:<25} {changed:>5} changed")

    assert 0 < changed <= 5000

assert (
    noisy["asset_age"]
    == 2014 - noisy["installation_year"]
).all()

print("\nAsset-age consistency: PASSED")
print("50% mixed-noise validation: PASSED")
PY


PYTHONPATH=src python -m team19_asset_data.overhead_conductors.run_masking \
  --rules MF01 \
  --input outputs/team19_shared/noisy_sources/N06_mixed_50/observed.csv \
  --config config/overhead_conductor_masking_rules.yaml \
  --output-dir outputs/team19_shared/masking_runs/noise_50


python - <<'PY'
import pandas as pd

base = "outputs/team19_shared/masking_runs"

reference = pd.read_csv(f"{base}/noise_0/masked_MF01.csv")
candidate = pd.read_csv(f"{base}/noise_50/masked_MF01.csv")

columns = [
    "installation_year",
    "distance_to_sea_km",
    "length_km",
    "voltage_category",
    "location_group",
    "corrosion_zone",
    "material",
    "covering",
]

assert len(reference) == len(candidate) == 10000
assert reference["segment_id"].equals(candidate["segment_id"])

print("=== 50% NOISE MASK VALIDATION ===")

for column in columns:
    reference_mask = reference[column].isna()
    candidate_mask = candidate[column].isna()

    missing = int(candidate_mask.sum())
    same_mask = reference_mask.equals(candidate_mask)

    print(
        f"{column:<25} "
        f"Missing: {missing:<5} "
        f"Same mask: {same_mask}"
    )

    assert missing == 2000
    assert same_mask, f"Mask mismatch: {column}"

assert reference["asset_age"].isna().equals(
    candidate["asset_age"].isna()
)

print("\n50% noise + 20% missingness: PASSED")
PY
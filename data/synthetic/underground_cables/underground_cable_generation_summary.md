# Underground Cable Synthetic Data Generation Summary

## Purpose

This dataset was created as a **fully synthetic underground-cable asset population** for Team 19's imputation and validation experiments. It is intended to provide a complete reference dataset from which known values can later be masked, imputed, and compared with the original ground truth.

No row should be interpreted as a real SA Power Networks asset record. The environmental variables are also synthetic proxies and are not measured South Australian groundwater or corrosion observations.

## Generation configuration

- **Asset class:** Underground cables
- **Rows in current dataset:** 5,000
- **Columns:** 21
- **Reference year used for asset age:** 2026
- **Generator default random seed:** 19
- **Generator default row count:** 5,000
- **Generation approach:** reproducible rule-based and probabilistic synthetic generation
- **Missingness at generation:** none; the reference dataset is intentionally complete
- **Intended next step:** controlled masking followed by explainable imputation and validation

## Schema

The dataset contains the following 21 fields:

`asset_id`, `feeder_id`, `rated_voltage_kv`, `installation_year`, `asset_age_years`, `cable_length_m`, `insulation_type`, `conductor_material`, `conductor_size_mm2`, `core_count`, `installation_method`, `latitude`, `longitude`, `joint_count`, `route_geometry_wkt`, `cable_type_raw`, `groundwater_depth_m`, `groundwater_exposure_score`, `groundwater_indicator`, `corrosion_exposure_score`, `corrosion_indicator`

The detailed rule, dependency, parameters, rationale and limitations for each field are recorded in `generation_assumptions.csv`.

## Core generation logic

1. A unique `asset_id` is assigned to every synthetic record and each cable is allocated to one of 60 non-uniform synthetic feeders.
2. Rated voltage is sampled as either 11 kV or 33 kV.
3. `installation_year` is sampled from a scaled Beta distribution between 1950 and 2026. `asset_age_years` is then derived directly from installation year.
4. Insulation type depends on installation year so older synthetic cables are more likely to use PILC and newer cables are more likely to use XLPE.
5. Conductor material depends on insulation and installation year, creating a deliberate but imperfect relationship between asset age, insulation and material.
6. Conductor size is sampled from material-specific size categories with an additional voltage-based constraint.
7. Core count is conditionally sampled from the voltage class.
8. Cable length follows a right-skewed lognormal distribution. Joint count then follows a Poisson process whose expected value increases with cable length.
9. Installation method is sampled as either Duct/Conduit or Direct Buried.
10. Synthetic latitude and longitude are generated around approximate South Australian population centres. A simple WKT route is constructed around each synthetic location.
11. `cable_type_raw` is derived from structured cable properties to create a realistic-looking raw descriptive field.
12. Groundwater depth is generated from a synthetic spatial function plus noise. Shallower depth leads to higher groundwater exposure.
13. Corrosion exposure is generated mainly from groundwater exposure with a smaller installation-method effect and additional noise.
14. Low/Medium/High environmental indicators are derived from the corresponding continuous exposure scores.

## Current dataset checks

### Age and physical variables

- Installation year range: **1950-2026**
- Asset age range: **0-76 years**
- Cable length range: **15.00-1800.00 m**
- Median cable length: **165.66 m**
- Joint-count range: **0-9**

### Categorical distributions

- Rated voltage: 11: 3849 (77.0%), 33: 1151 (23.0%)
- Insulation type: PILC: 1280 (25.6%), XLPE: 3720 (74.4%)
- Conductor material: Aluminium: 3183 (63.7%), Copper: 1817 (36.3%)
- Core count: 1: 3077 (61.5%), 3: 1923 (38.5%)
- Installation method: Direct Buried: 1167 (23.3%), Duct/Conduit: 3833 (76.7%)

### Environmental synthetic proxies

- Groundwater depth range: **1.16-11.43 m**
- Groundwater exposure score range: **0.240-0.943**
- Groundwater indicator: High: 624 (12.5%), Low: 173 (3.5%), Medium: 4203 (84.1%)
- Corrosion exposure score range: **0.121-0.984**
- Corrosion indicator: High: 495 (9.9%), Low: 402 (8.0%), Medium: 4103 (82.1%)

## Reproducibility

The generator accepts a row count, random seed and output path. The default configuration is:

```bash
python generate_underground_cable_synthetic.py \
    --rows 5000 \
    --seed 19 \
    --output Underground_Cable_Synthetic_data.csv
```

Using a fixed seed makes the probabilistic generation reproducible. The environmental proxy functions additionally use deterministic per-asset random streams so their noise remains stable for a given asset identifier, label and seed.

## Important dependency and leakage notes

Several variables are intentionally related. These relationships are useful for testing whether an imputation method can recover information from realistic predictor structure, but they also create potential target leakage.

The clearest example is `asset_age_years`, which is calculated directly from `installation_year`. If `installation_year` is used as the imputation target, `asset_age_years` must be excluded or masked on the same rows. Likewise, `groundwater_indicator` is derived from `groundwater_exposure_score`, and `corrosion_indicator` is derived from `corrosion_exposure_score`; derived versions should not be used to make an imputation task artificially easy.

## Limitations

- The dataset is synthetic and is not calibrated against confidential SAPN asset-register distributions.
- Location points and routes are artificial and do not represent real network assets.
- Groundwater and corrosion fields are method-testing proxies rather than measured environmental data.
- The probabilities, thresholds and conditional relationships were selected to create plausible and explainable structure for experimentation; they are not engineering standards.
- The generator is suitable for controlled imputation benchmarking, but results obtained on this data must later be checked against authorised real data when available.

## Intended validation use

The complete dataset should be preserved as the reference ground truth. For imputation experiments, selected known values should be deliberately masked using a reproducible masking strategy. The baseline and alternative imputation methods can then be evaluated against the hidden original values, including recovery error, distribution stability, confidence/uncertainty and downstream model stability.

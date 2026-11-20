# Literature Basis for Relationship Validation

These sources were reviewed before implementing the overhead-conductor audit.
They inform the experiment design; they do not turn the synthetic data into
SAPN data.

| Source | Relevance to this ticket |
|---|---|
| Troyanskaya et al. (2001), [Missing value estimation methods for DNA microarrays](https://www.cs.princeton.edu/courses/archive/fall03/cs597F/Articles/missing-value-estimation.pdf) | Supports testing KNN-style methods with explicit attention to scaling, missingness rate and neighbourhood size. |
| Stekhoven and Buhlmann (2012), [MissForest: non-parametric missing value imputation for mixed-type data](https://academic.oup.com/bioinformatics/article/28/1/112/219101) | Supports treating mixed categorical/numeric predictors and nonlinear relationships as method-design concerns. |
| van Buuren and Groothuis-Oudshoorn (2011), [mice: Multivariate Imputation by Chained Equations in R](https://www.jstatsoft.org/article/view/v045i03) | Supports documenting the missingness process and uncertainty rather than reporting a single accuracy number only. |
| Kaufman et al. (2012), [Leakage in Data Mining: Formulation, Detection, and Avoidance](https://cris.tau.ac.il/en/publications/leakage-in-data-mining-formulation-detection-and-avoidance/) | Supports treating target-revealing fields and train/evaluation contamination as leakage risks, not relying on Pearson correlation alone. |
| Kuhnel et al. (2024), [Synthetic data generation for a longitudinal cohort study](https://www.nature.com/articles/s41598-024-62102-2) | Supports checking distributions, association structures, deterministic dependencies and practical utility when validating synthetic data. |

## Design decisions carried into the code

- Use Cramer's V and normalised mutual information for nominal multiclass
  relationships instead of encoding material as arbitrary numbers.
- Report support counts and apply a minimum-support threshold to category purity.
- Keep direct target-revealing `conductor_type` in the complete register only;
  exclude it from model-ready data and flag it in the leakage audit.
- Treat `installation_year` and `asset_age` as duplicate evidence because age
  is deterministically derived from the installation year.
- Repeat the audit across relationship-strength profiles and independent seeds
  before using the data for imputation comparisons.

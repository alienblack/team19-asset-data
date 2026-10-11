from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency


BASE_DIR = Path("data/synthetic/overhead_conductors")

STRENGTHS = list(range(0, 101, 10))


def cramers_v(x: pd.Series, y: pd.Series) -> float:
    """
    Calculate bias-corrected Cramér's V for two categorical variables.
    """
    confusion_matrix = pd.crosstab(x, y)

    if confusion_matrix.empty:
        return np.nan

    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.to_numpy().sum()

    if n <= 1:
        return np.nan

    phi2 = chi2 / n
    r, k = confusion_matrix.shape

    phi2_corrected = max(
        0,
        phi2 - ((k - 1) * (r - 1)) / (n - 1)
    )

    r_corrected = r - ((r - 1) ** 2) / (n - 1)
    k_corrected = k - ((k - 1) ** 2) / (n - 1)

    denominator = min(k_corrected - 1, r_corrected - 1)

    if denominator <= 0:
        return 0.0

    return np.sqrt(phi2_corrected / denominator)


def create_installation_era(df: pd.DataFrame) -> pd.Series:
    """
    Convert installation_year into the same broad eras used
    by the synthetic generator.
    """
    return pd.cut(
        df["installation_year"],
        bins=[-np.inf, 1949, 1977, 1994, np.inf],
        labels=[
            "pre_1950",
            "1950_1977",
            "1978_1994",
            "from_1995",
        ],
    )


def analyse_strength(strength: int) -> dict:
    file_path = (
        BASE_DIR
        / f"strength_{strength}"
        / "material_model_ready_data.csv"
    )

    df = pd.read_csv(file_path)

    df["installation_era"] = create_installation_era(df)

    return {
        "strength": strength,

        "installation_year_to_voltage": cramers_v(
            df["installation_era"],
            df["voltage_category"],
        ),

        "voltage_to_location": cramers_v(
            df["voltage_category"],
            df["location_group"],
        ),

        "voltage_to_corrosion": cramers_v(
            df["voltage_category"],
            df["corrosion_zone"],
        ),

        "location_to_corrosion": cramers_v(
            df["location_group"],
            df["corrosion_zone"],
        ),

        "installation_year_to_material": cramers_v(
            df["installation_era"],
            df["material"],
        ),

        "voltage_to_material": cramers_v(
            df["voltage_category"],
            df["material"],
        ),

        "corrosion_to_material": cramers_v(
            df["corrosion_zone"],
            df["material"],
        ),
    }


def main():
    results = []

    for strength in STRENGTHS:
        print(f"Analysing strength {strength}%...")
        results.append(analyse_strength(strength))

    results_df = pd.DataFrame(results)

    print("\nRelationship strength validation:")
    print(results_df.to_string(index=False))

    # Save numeric results
    output_dir = BASE_DIR / "relationship_validation"
    output_dir.mkdir(parents=True, exist_ok=True)

    results_df.to_csv(
        output_dir / "relationship_strength_validation.csv",
        index=False,
    )

    # Plot
    relationship_columns = [
        col
        for col in results_df.columns
        if col != "strength"
    ]

    plt.figure(figsize=(12, 7))

    for relationship in relationship_columns:
        plt.plot(
            results_df["strength"],
            results_df[relationship],
            marker="o",
            label=relationship.replace("_", " "),
        )

    plt.xlabel("Configured relationship strength (%)")
    plt.ylabel("Measured association (Cramér's V)")
    plt.title(
        "Configured Relationship Strength vs Measured Association"
    )

    plt.xticks(STRENGTHS)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    output_path = (
        output_dir
        / "relationship_strength_validation.png"
    )

    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"\nSaved results to: {output_dir}")


if __name__ == "__main__":
    main()
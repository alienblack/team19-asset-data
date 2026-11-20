"""The parts of Subteam A's ``data.py`` that their model files import, copied unchanged."""
from dataclasses import dataclass
import numpy as np
import pandas as pd

SNAPSHOT_YEAR = 2014


@dataclass(frozen=True)
class Schema:
    """Describe column names and categorical codes for a transformer or overhead asset matrix."""
    names: tuple
    categories: dict  # column index -> fixed domain labels, in integer-code order
    transformer: bool = True

    @property
    def numeric(self):
        """Return the indexes of numeric columns in schema order."""
        return [j for j in range(len(self.names)) if j not in self.categories]

    def i(self, name):
        """Return the column index for the supplied field name."""
        return self.names.index(name)

    def decode(self, array):
        """Convert an encoded matrix to a named DataFrame with the original category labels."""
        df = pd.DataFrame(array, columns=self.names)
        for j, levels in self.categories.items():
            df[self.names[j]] = [levels[int(x)] if np.isfinite(x) else None for x in array[:, j]]
        return df


FIELD_GROUPS = {
    'location': ('x', 'y', 'altitude'),
    'gases': ('measured_gas_a', 'measured_gas_b', 'measured_gas_c', 'total_gas'),
    'environment': ('measured_humid', 'measured_salt', 'measured_overgrowth'),
    'date/model': ('install_date', 'model_id'),
}


def share_group_draws(draws, schema, groups):
    """Copy the row-by-field random draws and reuse each group’s first draw across its members."""
    shared = draws.copy()
    for members in groups:
        columns = [schema.i(c) for c in members]
        shared[:, columns] = shared[:, columns[:1]]
    return shared

"""Model names shown in tables and figures.

Fitting and saved tasks use the names in ``models``. The evaluation renames a model where a
shorter or more common name exists, so every table and figure uses the same name as the text.
"""

from .models import CATEGORY_METHODS as FITTED_CATEGORY_METHODS
from .models import NUMERIC_METHODS as FITTED_NUMERIC_METHODS

SHOWN_NAMES = {"Gaussian EM": "GMM"}


def shown(method):
    """The name shown in tables and figures for a model fitted under ``method``."""
    return SHOWN_NAMES.get(method, method)


NUMERIC_METHODS = tuple(shown(m) for m in FITTED_NUMERIC_METHODS)
CATEGORY_METHODS = tuple(shown(m) for m in FITTED_CATEGORY_METHODS)

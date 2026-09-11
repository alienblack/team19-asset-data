"""Default configuration for the Member 3 validation framework.

Leave values empty when using command-line arguments.  Paths and column names are
validated by the runner before any analysis is attempted.
"""

REFERENCE_FILE = ""
MASKED_FILE = ""
BASELINE_FILE = ""
ALTERNATIVE_FILE = ""

TARGET_COLUMN = ""
ID_COLUMN = ""

OUTPUT_DIR = "member3_validation/outputs"

# Preliminary project rule, not an SAPN standard.  An imputed value is left as
# TBD unless the caller supplies a supported confidence label.
DEFAULT_IMPUTED_CONFIDENCE = "TBD"


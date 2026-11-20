"""Model code from the other two Team 19 branches, kept as written apart from import paths.

- ``subteam_a``: Subteam A's benchmark (Rishank Goyal, SAPN_endtoend), files ``models.py``,
  ``mixture.py`` and ``features.py`` copied from ``src/subteam_a``. ``data.py`` holds only the
  parts of their ``data.py`` that these files import.
- ``pipeline``: the team pipeline, files ``predictive_missforest.py``, ``postprocess.py`` and
  ``simple_imputer.py`` from ``src/imputation`` and ``comparison_models.py`` from
  ``experiment/transformer/scripts``. Only relative imports were changed.

``models_external.py`` adapts these classes to the study's task interface.
"""

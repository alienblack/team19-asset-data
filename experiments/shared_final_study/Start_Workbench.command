#!/bin/sh
cd "$(dirname "$0")" || exit 1
if [ ! -x .venv/bin/python ]; then
  echo "Create the Python 3.12 environment first. Follow START_HERE.md."
  echo "Press Return to close."
  read -r launcher_key
  exit 1
fi
exec .venv/bin/python run_study.py --mode guided

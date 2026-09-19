$ErrorActionPreference = "Stop"
python run_all.py --device auto --quick
python -m pytest

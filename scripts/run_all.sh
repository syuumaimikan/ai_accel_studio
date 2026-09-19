#!/usr/bin/env bash
set -euo pipefail
python run_all.py --device auto --quick
python -m pytest

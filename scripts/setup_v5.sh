#!/usr/bin/env bash
set -euo pipefail
python -m pip install -U pip setuptools wheel
python -m pip install -r requirements-full.txt
python -m ai_accel_lab doctor

$ErrorActionPreference = "Stop"
New-Item -ItemType Directory -Force -Path results | Out-Null
python -m ai_accel_lab linear --device cuda --preset medium --repeats 200 --power --csv results/linear_cuda.csv
python -m ai_accel_lab transformer --device cuda --preset base --repeats 100 --power --csv results/transformer_cuda.csv
python -m ai_accel_lab sweep --device cuda --preset medium --out results/pareto_cuda

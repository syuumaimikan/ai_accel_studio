# AI Acceleration Studio v5.0.0

既存の Hugging Face decoder-only LLM を、**速度 / J-token / VRAM / 品質を同じ条件で測定しながら最適化する研究・実装用Studio**です。

v5は、v4実測で分かった次の問題を直接狙っています。

- packed INT2 GEMV自体は改善したがFP16よりまだ遅い
- custom runtime全体ではkernel launch / Python dispatchがさらに効いていた
- 1つのcalibration集合ではAccuracy Guardが過学習することがあった
- INT2が厳しい層を全てFP16へ戻すと圧縮率を失う
- SmolLMのように全層rollbackした場合でも、custom経路の測定がbaselineより遅く見えることがあった
- `hf-kernels-auto` / TorchAOのoptional dependency差で比較行が失敗することがあった

## v5の主要変更

### 1. INT2 -> INT4 -> FP16 mixed-bit planner

各Linearを最初から同じbit数に固定しません。

```text
INT2 + activation-aware group scale
  -> exact FP16 outlier columns
  -> optional low-rank residual
      | pass
      v
     INT2
      |
      | fail
      v
packed INT4 + exact outlier columns
      | pass
      v
     INT4
      |
      | fail
      v
     FP16
```

精度の高い層だけ4/16bitへ昇格させ、帯域削減が効く層は2bitのまま残します。

### 2. held-out Robust Accuracy Guard

v4は同じcalibration集合で計画と最終判定を行うため、局所的に良くても別テキストでPPLが悪化する場合がありました。

v5は入力集合を交互に分け、

- plan/calibration prompts
- held-out validation prompts

として使用します。最終rollback条件は平均だけでなく:

- mean relative NLL increase
- P95 relative NLL increase
- worst relative NLL increase

を見ます。

### 3. Static KV Cache + torch.compile / CUDA-Graph-friendly runtime

`baseline-static-compile` と `custom-mixedbit-v5-graph` を追加しました。

```text
Static KV Cache
    +
torch.compile(mode="reduce-overhead")
    +
graph-safe custom INT2/INT4 decode modules
```

固定shapeのdecodeでPython/C++/CUDA-driver launch setupを減らすための経路です。

### 4. M<=4 decode専用 packed INT2 / INT4

- `decode-int2-v4`: 4 weights / byte
- `decode-int4-v5`: 2 weights / byte

どちらも小M autoregressive decode向けです。packed weightをHBMから読み、kernel内でdecode + group scaleを行います。

INT2はnative INT2 Tensor Coreを名乗るものではなく、**2-bit storage + fused dequant/GEMV**です。

### 5. zero-optimized-layer exact baseline reload

Robust Guardが全層をFP16へ戻した場合、custom wrapperやallocatorの残留状態でbaseline-equivalent measurementが遅く見えないよう、**モデルをbaselineとして再ロードしてから測定**します。

### 6. multi-text perplexity

GUIのPPL評価も1文章だけではなく `---` 区切りの複数held-out文章をtoken-weighted NLLで集約します。

### 7. Dependency Doctor

```bash
python -m ai_accel_lab doctor
```

インストール済みpackage metadataを読み、特に:

- `transformers -> kernels`
- `torchao -> mslk`

の要求範囲を表示します。固定versionを憶測で案内しません。

## 比較方式

- `baseline-fp16`
- `baseline-static-compile`
- `hf-kernels-auto`
- `sdpa`
- FlashAttention 2 / 3 / paged FA3
- `torch-compile`
- TorchAO INT4 / INT8 / INT2
- native 2:4 sparse
- custom INT2 v4系
- `custom-mixedbit-v5`
- `custom-mixedbit-v5-graph`
- Hopper WGMMA/TMA experiments
- Blackwell tcgen05/TMEM / warp-specialization experiments
- QKV+RoPE+KV-cache+attention fusion microbenchmark

## セットアップ

Python 3.10+。まずGPU/driverに合ったCUDA版PyTorchを導入してください。

```bash
pip install -r requirements-full.txt
```

依存関係を確認:

```bash
python -m ai_accel_lab doctor
```

GPU経路を検証:

```bash
python -m ai_accel_lab selftest --microbench --json results/selftest.json
```

## GUI

```bash
python studio.py
```

または:

```bash
python -m ai_accel_lab studio
```

## Qwen2.5-0.5B 推奨比較

```bash
python -m ai_accel_lab model-bench \
  --models Qwen/Qwen2.5-0.5B-Instruct \
  --methods baseline-fp16,baseline-static-compile,hf-kernels-auto,torchao-int4,custom-mixedbit-v5,custom-mixedbit-v5-graph \
  --new-tokens 64 \
  --group-size 128 \
  --nmse-limit 0.06 \
  --cosine-limit 0.96 \
  --global-loss-budget 0.02 \
  --residual-rank 8
```

GUIではより多様な16 calibration promptsと4 held-out perplexity textsを標準で入れています。

## Decode kernel単体比較

```bash
python -m ai_accel_lab kernel-suite \
  --m 1 \
  --n 4096 \
  --k 4096 \
  --group-size 64 \
  --repeats 500 \
  --power
```

GUIでは以下を同時比較できます。

```text
torch-fp16
decode-int2-v4
decode-int4-v5
groupwise-int2
native-2to4
...
```

## 判断ルール

Studioは低bit/sparse/persistentという名前だけで高速と判定しません。

モデルごとにbaselineと比較して:

- decode tokens/s
- TTFT
- total tokens/s
- W
- J/generated-token
- model storage
- current/reserved/peak VRAM
- perplexity
- quality pass

を測り、品質条件を通った実測候補の中から最速をマークします。

小型モデルではFP16 + static cache + compileが勝つことがあります。それも正常な結果です。大型モデルではweight-bandwidth比率が上がるためmixed-bitの価値が増える可能性があります。

## Hopper / Blackwell

実験経路を維持しています。

- Hopper WGMMA
- TMA
- persistent scheduling
- Blackwell tcgen05 / Tensor Memory
- warp specialization
- PTX/SASS marker verification

ただしTriton公式の例でもshapeによりcuBLASがpersistent kernelを上回るため、自動的に高度kernelを勝者扱いしません。

## Bundle

GUIの **Robust Mixed-Bit Guard v5** からv5 bundleを保存できます。

v5 bundleは置換済み層についてdense source weightをstate_dictに保持せず、packed INT2/INT4、scale、outlier、必要なresidualだけを保存します。

## テスト

```bash
pytest
```

CPU-only環境ではCUDA専用testはskipされます。

## 主な構成

```text
ai_accel_lab/
  kernels/
    decode_int2.py
    groupwise_int2.py
    groupwise_int4.py       # v5
    tma_tensorcore.py
    gluon_hopper_blackwell.py
    gluon_warp_specialized.py
  optimize/
    hybrid_int2.py
    hybrid_int4.py          # v5
    planner_v5.py           # 2 -> 4 -> 16 bit planner
    robust_guard.py         # held-out mean/P95/worst NLL guard
    bundle.py
  runtime_v5.py             # StaticCache + compile benchmark path
  calibration_suite.py
  doctor.py
  model_benchmark.py
  studio/app.py
```

## 注意

このリポジトリは研究・検証用です。特定GPU/shapeでの速度倍率は実機測定なしには保証しません。custom Hopper/Blackwell kernelはself-testに通った経路のみ使用してください。

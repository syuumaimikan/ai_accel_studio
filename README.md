# AI Acceleration Studio v3.0.0

既存の Hugging Face / PyTorch モデルへ高速化を適用し、**速度・消費電力・VRAM・出力品質を同じGUIで比較**するための研究/実装プロジェクトです。

v3 は単なる fake quant benchmark ではなく、packed INT2、2:4 semi-structured sparsity、Hopper/Blackwell専用経路、Accuracy Guard、モデルbundle保存まで含みます。

## 重要な前提

高速化・省電力化・精度維持はGPU、shape、モデル、context lengthで結果が変わります。**全モデルで必ず元モデルより高速・省電力・高精度になることは保証できません。**

このStudioは、候補手法を実測し、Accuracy Guardで精度劣化が大きい層をFP16へ戻すことで、性能と品質のPareto点を探す設計です。

## 主な高速化経路

### 1. Groupwise INT2

- 4 weights / byte の実packed storage
- group size: 32 / 64 / 128 / 256
- kernel内 unpack + dequant + matmul
- activation-aware scale refinement
- low-rank residual correction
- decode向け persistent INT2 kernel

### 2. 2:4 Semi-Structured Sparsity

`torch.sparse.to_sparse_semi_structured` / vendor sparse GEMM経路を使います。対応GPUではSparse Tensor Coreの実装へdispatchされます。

### 3. Hopper / Blackwell

`ai_accel_lab/kernels/gluon_hopper_blackwell.py` は起動時self-testに成功した環境でのみ有効になります。

- Hopper SM90: TMA + asynchronous WGMMA
- Blackwell SM100+: TMA + tcgen05 + Tensor Memory
- persistent tile scheduling
- 3〜4段operand buffering
- async load / MMA overlap

`gluon_warp_specialized.py` にはBlackwell向けのload / MMA / store warp partition版もあります。

### 4. TMA

`tma_tensorcore.py` は `tl.make_tensor_descriptor` を使います。Hopper以降ではTritonがTMA-backed descriptor load/storeへlowerできる経路です。

### 5. Giant Decode Fusion

`fused_decode_attention.py` の実験kernelはsingle-token decodeで以下を1 Triton kernel内にまとめます。

```text
packed groupwise INT2 Q/K/V projection
                ↓
               RoPE
                ↓
          KV cache append
                ↓
       causal online-softmax
                ↓
          attention output
```

現時点のgiant fusionは安全のため **MHA + head_dim 64/128** に限定しています。GQA/MQAや未対応shapeは通常経路へfallbackします。

## Accuracy Guard

量子化を全層へ強制すると品質が落ちるモデルがあります。そのためStudioは代表inputを記録し、各Linearについて次を試します。

```text
INT2
INT2 + rank 4 residual
INT2 + rank 8 residual
INT2 + rank 16 residual
...
FP16 fallback
```

判定にはoutput NMSEとcosine similarityを使います。基準を満たせない敏感層はFP16のまま残します。

つまり「INT2だから速い」を優先するのではなく、**品質制約の中で最も軽い構成**を狙います。

## GUI

### インストール

Linux / WSL2 + NVIDIA GPU 推奨です。

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements-full.txt
```

CUDA対応PyTorchは使用するCUDA環境に合わせて先にインストールしてください。

### 起動

```bash
python studio.py
```

または

```bash
python -m ai_accel_lab studio
```

ブラウザで `http://127.0.0.1:7860` を開きます。

### GUIタブ

- **Hardware** — GPU世代、SM、TMA/WGMMA/tcgen05対応を表示
- **Kernel Lab** — FP16 / INT2 / persistent / 2:4 / TMA / Hopper-Blackwell backend比較
- **Model Matrix** — 複数モデル × 複数手法を順番に比較
- **Giant Fusion** — QKV+RoPE+Attention融合kernel比較
- **Accuracy Guard** — 既存モデルをキャリブレーションし再利用可能bundleへ変換
- **Architecture paths** — 現在有効なbackendの説明
- **Runs** — CSV/JSONで保存された過去結果

## Model Matrix の比較項目

- TTFT (ms)
- decode tokens/s
- total tokens/s
- average GPU power (W)
- Joules / generated token
- peak VRAM
- perplexity
- Accuracy Guardで最適化できた層数
- generated text
- baseline比 speedup / J-token削減率 / perplexity差
- 品質許容内で最速だった実測方式の `recommended_measured` マーク

モデルは同時にVRAMへ載せず、**1モデル・1手法ずつロード→測定→解放**します。

## 対応モデル

Linear replacementはモデル固有コードに依存しないため、`q_proj/k_proj/v_proj/o_proj` や `gate_proj/up_proj/down_proj` を持つ多くのdecoder-only Transformerで利用できます。

GUIの初期例:

- `Qwen/Qwen2.5-0.5B-Instruct`
- `HuggingFaceTB/SmolLM2-360M-Instruct`

Transformers側の標準/既存高速化（Hub kernels、SDPA、FlashAttention系）も比較対象に含めています。

任意のHugging Face IDまたはローカルpathを入力できます。モデル固有のcustom opが強い場合は、baseline/torchao側がcustom INT2より速いこともあります。

## 比較手法

Model Matrixには次があります。

```text
baseline-fp16
hf-kernels-auto
sdpa
hf-flash-attn2-kernel
hf-flash-attn3-kernel
flash-attention-2
flash-attention-3
paged-flash-attention-3
torch-compile
torchao-int4
torchao-int8
torchao-int2
native-2to4
custom-int2-fast
custom-int2-residual
custom-int2-accuracy-guard
```

## 最適化bundle

Accuracy Guardタブでbundleを作ると以下が保存されます。

```text
optimized_bundles/...
├─ accel_manifest.json
├─ optimized_state.pt
└─ tokenizer/
```

ロードAPI:

```python
from ai_accel_lab.optimize.bundle import load_bundle
model, tokenizer, manifest, missing, unexpected = load_bundle(
    "optimized_bundles/model-int2"
)
```

## CLI

ハードウェア確認:

```bash
python -m ai_accel_lab hardware
```

高度backendのcorrectness self-test（H100/B100/B200上で推奨）:

```bash
python -m ai_accel_lab selftest --microbench --json results/selftest.json
```

kernel比較:

```bash
python -m ai_accel_lab kernel-suite \
  --m 32 --n 4096 --k 4096 \
  --group-size 128 --repeats 200 --power
```

モデル比較:

```bash
python -m ai_accel_lab model-bench \
  --models Qwen/Qwen2.5-0.5B-Instruct,HuggingFaceTB/SmolLM2-360M-Instruct \
  --methods baseline-fp16,torchao-int4,custom-int2-accuracy-guard
```

## 実際にWGMMA / MMA.SP / TMAが出たか確認

高水準API名だけで判断せず、生成codeを確認するためのscannerを入れています。

```bash
python tools/verify_codegen.py path/to/kernel.cubin
```

検出対象:

```text
WGMMA / wgmma.mma_async
MMA.SP / mma.sp
CPASYNC.BULK.TENSOR / TMA
TCGEN05 / tcgen05.mma
```

実プロファイルには Nsight Compute も推奨です。

## Self-test と fallback

高度backendは次の順で扱います。

```text
architecture check
      ↓
small correctness self-test
      ↓ pass
backend enabled
      ↓ fail
TMA / Triton / PyTorch fallback
```

高速なkernelでも誤差が許容範囲を超えれば有効化しない設計です。

## Tests

```bash
pytest -q
```

CUDA環境でのみ動くテストはCUDAなし環境ではskipされます。

## 現在の制約

- 手元の生成環境にはNVIDIA GPU/NVCC/Triton GPU runtimeがないため、Hopper/Blackwell専用kernelをこのZIP作成時に実機benchmarkできていません。
- CPUで実行できるpacking、Accuracy Guard、GUI import、既存benchmarkのテストは実行しています。
- Gluonはexperimental APIなのでTritonの将来版でAPI変更が起きる可能性があります。そのためself-test/fallbackを必須にしています。
- giant fusionはまずsingle-token MHA decodeへ対象を絞っています。モデル統合では未対応attention shapeを自動fallbackさせるのが安全です。

## 参考にした公式仕様

- NVIDIA PTX ISA: https://docs.nvidia.com/cuda/parallel-thread-execution/
- CUDA Programming Guide: https://docs.nvidia.com/cuda/cuda-programming-guide/
- Triton TMA tensor descriptors: https://triton-lang.org/main/python-api/generated/triton.language.make_tensor_descriptor.html
- Triton Persistent Kernels: https://triton-lang.org/main/getting-started/tutorials/gluon/persistence.html
- Triton Warp Specialization: https://triton-lang.org/main/getting-started/tutorials/gluon/warp-specialization.html
- Triton WGMMA: https://triton-lang.org/main/getting-started/tutorials/gluon/wgmma.html
- Triton Blackwell tcgen05: https://triton-lang.org/main/getting-started/tutorials/gluon/tcgen05.html
- PyTorch 2:4 semi-structured sparsity: https://docs.pytorch.org/tutorials/advanced/semi_structured_sparse.html
- torchao quantization: https://docs.pytorch.org/ao/stable/

# 実験設計

## A. Ternary residual rank
`rank = 0,2,4,8,16,32` を比較し、NMSE / weight bytes / latencyのPareto frontを見る。

## B. Activation-aware residual
通常SVD residualとactivation-aware residualを同rankで比較する。評価用入力はcalibration入力とは別にする。

## C. Structured sparsity
- 1:4
- 2:4
- 4:8

PyTorch dense fallbackでは実速度が出ない可能性があるため、MAC reductionと実測を分離する。

## D. Block sparsity
block size 16/32/64、keep 12.5/25/50/75%。GPUではper-sample gatherよりbatch-shared block selectionが有利か確認する。

## E. Delta
変更率 1/2/5/10/20%。threshold>0では近似誤差と速度のParetoを見る。

## F. Hierarchical memory
memory 4K/16K/64K。top blocks 1/2/4/8。full attentionとのcosineと速度を比較する。

## G. Analog/CIM simulation
bits 2/4/8、noise std 0/0.5/1/2/5%、digital correction rank 0/4/8/16。

## H. Real LLM
同一prompt corpusで baseline perplexityと置換後perplexityを比較する。速度評価は専用kernel実装後に行う。

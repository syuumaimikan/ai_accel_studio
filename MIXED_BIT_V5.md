# Mixed-bit v5

The goal is not to maximize the number of 2-bit layers. The goal is to minimize runtime memory traffic while meeting model-level quality constraints.

Per target Linear:

1. search activation-aware groupwise INT2 candidates;
2. optionally retain a small set of exact FP16 input columns;
3. use low-rank residual only where needed;
4. if INT2 still misses the local quality bound, try packed INT4;
5. keep FP16 only when both fail.

After local planning, a held-out model-level guard evaluates per-text NLL degradation and can restore risky layers.

The final report includes counts for INT2, INT4 and FP16 layers.

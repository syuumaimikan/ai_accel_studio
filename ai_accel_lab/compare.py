from __future__ import annotations

import math
import pandas as pd


def annotate_model_matrix(df: pd.DataFrame, max_perplexity_rel_increase: float = 0.03) -> pd.DataFrame:
    """Add baseline-relative metrics and a measured best-under-quality marker.

    Selection is deliberately local to each model and only uses successful measured rows.
    A candidate must keep perplexity within the configured relative increase when both
    baseline and candidate perplexity are available. Among passing candidates, the
    highest decode throughput wins; J/token is the tie-breaker when available.
    """
    if df.empty or 'model' not in df.columns:
        return df
    out = df.copy()
    out['speedup_vs_baseline'] = float('nan')
    out['energy_reduction_vs_baseline'] = float('nan')
    out['perplexity_delta'] = float('nan')
    out['quality_pass'] = False
    out['recommended_measured'] = False

    for model, idx in out.groupby('model').groups.items():
        sub = out.loc[idx]
        base_rows = sub[sub['method'].isin(['baseline-fp16','baseline-bf16'])]
        if base_rows.empty:
            continue
        base = base_rows.iloc[0]
        base_tps = float(base.get('decode_tokens_s') or 0.0)
        base_energy = base.get('joules_per_generated_token')
        base_ppl = base.get('perplexity')

        for ridx, row in sub.iterrows():
            if row.get('error') not in (None, '', float('nan')) and pd.notna(row.get('error')):
                continue
            tps = row.get('decode_tokens_s')
            if pd.notna(tps) and base_tps > 0:
                out.at[ridx, 'speedup_vs_baseline'] = float(tps) / base_tps
            eng = row.get('joules_per_generated_token')
            if pd.notna(eng) and pd.notna(base_energy) and float(base_energy) > 0:
                out.at[ridx, 'energy_reduction_vs_baseline'] = 1.0 - float(eng) / float(base_energy)
            ppl = row.get('perplexity')
            if pd.notna(ppl) and pd.notna(base_ppl) and float(base_ppl) > 0:
                rel = float(ppl) / float(base_ppl) - 1.0
                out.at[ridx, 'perplexity_delta'] = float(ppl) - float(base_ppl)
                out.at[ridx, 'quality_pass'] = rel <= max_perplexity_rel_increase
            else:
                # If no perplexity was requested, do not invent a quality result;
                # baseline itself is safe, candidates remain unqualified for auto-pick.
                out.at[ridx, 'quality_pass'] = row.get('method') in ('baseline-fp16','baseline-bf16')

        eligible = out.loc[idx]
        eligible = eligible[(eligible['quality_pass']) & eligible['decode_tokens_s'].notna()]
        eligible = eligible[eligible['decode_tokens_s'] > 0]
        if eligible.empty:
            continue
        # Max throughput first, lower energy as deterministic tie breaker.
        e = eligible.copy()
        e['_energy_sort'] = e['joules_per_generated_token'].fillna(float('inf'))
        best_idx = e.sort_values(['decode_tokens_s','_energy_sort'], ascending=[False,True]).index[0]
        out.at[best_idx, 'recommended_measured'] = True
    return out

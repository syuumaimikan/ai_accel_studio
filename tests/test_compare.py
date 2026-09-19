import pandas as pd
from ai_accel_lab.compare import annotate_model_matrix


def test_measured_recommendation_respects_quality():
    df = pd.DataFrame([
        {'model':'m','method':'baseline-fp16','decode_tokens_s':10.0,'joules_per_generated_token':2.0,'perplexity':10.0,'error':None},
        {'model':'m','method':'fast-good','decode_tokens_s':20.0,'joules_per_generated_token':1.0,'perplexity':10.2,'error':None},
        {'model':'m','method':'fast-bad','decode_tokens_s':30.0,'joules_per_generated_token':0.8,'perplexity':12.0,'error':None},
    ])
    out = annotate_model_matrix(df, max_perplexity_rel_increase=0.03)
    good = out[out.method=='fast-good'].iloc[0]
    bad = out[out.method=='fast-bad'].iloc[0]
    assert good.recommended_measured
    assert good.speedup_vs_baseline == 2.0
    assert good.energy_reduction_vs_baseline == 0.5
    assert not bad.quality_pass

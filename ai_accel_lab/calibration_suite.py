from __future__ import annotations

DEFAULT_CALIBRATION_TEXTS = [
    'Explain how GPU memory bandwidth affects autoregressive language-model inference.',
    'Write a Python function that returns the first n Fibonacci numbers and explain its complexity.',
    'If a train travels 120 km in 80 minutes, compute its average speed in km/h.',
    'Summarize the trade-off between quantization error and memory bandwidth in neural networks.',
    '次の文章を簡潔に要約してください。大規模言語モデルの推論では、計算性能だけでなくメモリ帯域幅も重要である。',
    '日本語で、行列積がニューラルネットワークでどのように使われるか説明してください。',
    'Compare a hash table and a balanced binary search tree for lookup-heavy workloads.',
    'Given A implies B and B implies C, explain what follows about A and C.',
    'Write a short C++ loop that sums the elements of a vector without changing the vector.',
    'Explain the difference between latency, throughput, and energy per generated token.',
    'Translate into Japanese: Efficient inference depends on both arithmetic throughput and data movement.',
    'A model predicts probabilities 0.7, 0.2, and 0.1. Explain what cross entropy measures for the correct first class.',
    'Describe one advantage and one disadvantage of using a static KV cache during language-model generation.',
    'Explain why a lower-bit weight format can save memory but still run slower if dequantization overhead is too high.',
    'What is rotary positional embedding, and why is it applied to queries and keys?',
    'Give a concise explanation of grouped-query attention and how it changes the KV cache size.',
]

DEFAULT_EVAL_TEXTS = [
    'Efficient language model inference requires balancing arithmetic throughput, memory bandwidth, cache behavior, and kernel launch overhead.',
    'Quantization reduces model storage and memory traffic, but numerical error and dequantization cost can offset the theoretical performance gain.',
    'Autoregressive decoding repeatedly executes transformer layers for one or a few new tokens, making low latency and efficient key-value caching important.',
    'A robust optimization should be evaluated on held-out text rather than only the calibration examples used to choose quantization parameters.',
]

def joined_calibration(): return '\n---\n'.join(DEFAULT_CALIBRATION_TEXTS)
def joined_eval(): return '\n---\n'.join(DEFAULT_EVAL_TEXTS)

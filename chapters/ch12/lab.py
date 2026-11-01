#!/usr/bin/env python3
"""12장 CPU 개념 실습. 표준 라이브러리만 사용하며 실제 LLM은 학습하지 않는다.

LoRA 행렬, 균일 양자화, 마스크가 있는 SFT 손실, 쌍별 보상 손실,
두 응답으로 제한된 DPO 정책의 경사하강을 실제 계산한다.
"""
import argparse
import json
import math
import random


def matmul(a, b):
    if len(a[0]) != len(b):
        raise ValueError('행렬의 안쪽 차원이 일치해야 합니다.')
    return [[sum(a[i][k] * b[k][j] for k in range(len(b)))
             for j in range(len(b[0]))] for i in range(len(a))]


def matvec(a, x):
    return [sum(v * w for v, w in zip(row, x)) for row in a]


def sigmoid(x):
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def softplus(x):
    return max(x, 0.0) + math.log1p(math.exp(-abs(x)))


def logit(p):
    return math.log(p) - math.log1p(-p)


def mse(a, b):
    return sum((x-y)**2 for x, y in zip(a, b)) / len(a)


def quantize_uniform(values, bits):
    """학습용 min-max 균일 양자화. NF4나 bitsandbytes 구현이 아니다."""
    lo, hi = min(values), max(values)
    if lo == hi:
        return list(values)
    levels = 2**bits - 1
    step = (hi - lo) / levels
    return [lo + round((v-lo)/step) * step for v in values]


def lora_demo(rank):
    rng = random.Random(12)
    m, n, alpha = 4, 6, 4.0
    base = [[rng.uniform(-1, 1) for _ in range(n)] for _ in range(m)]
    a = [[rng.uniform(-0.2, 0.2) for _ in range(n)] for _ in range(rank)]
    b = [[rng.uniform(-0.2, 0.2) for _ in range(rank)] for _ in range(m)]
    x = [1.0, -1.0, 0.5, 0.0, 2.0, -0.5]
    delta = matmul(b, a)
    scale = alpha / rank
    merged = [[base[i][j] + scale * delta[i][j] for j in range(n)]
              for i in range(m)]
    direct = matvec(merged, x)
    base_y, adapter_y = matvec(base, x), matvec(b, matvec(a, x))
    split = [u + scale*v for u, v in zip(base_y, adapter_y)]
    max_error = max(abs(u-v) for u, v in zip(direct, split))
    assert max_error < 1e-12, '병합과 두 경로 계산은 같은 선형 변환이어야 함'
    return {'shape': [m, n], 'rank': rank, 'alpha': alpha,
            'full_parameters': m*n, 'lora_parameters': rank*(m+n),
            'scale': scale, 'merged_output': direct,
            'split_output': split, 'max_error': max_error,
            'large_example_full': 12288**2,
            'large_example_rank8': 8*(12288+12288)}


def quantization_demo(bits):
    values = [-0.31, -0.12, 0.08, 0.27, -0.4, 0.1, 0.3, 8.0]
    global_q = quantize_uniform(values, bits)
    block_q = []
    for i in range(0, len(values), 4):
        block_q.extend(quantize_uniform(values[i:i+4], bits))
    return {'bits': bits, 'levels': 2**bits, 'values': values,
            'global_reconstructed': global_q, 'block_reconstructed': block_q,
            'global_mse': mse(values, global_q), 'block_mse': mse(values, block_q),
            'note': '균일 min-max 양자화; 블록마다 min/max 저장 비용은 MSE에 포함하지 않음'}


def sft_demo():
    tokens = ['<user>', '질문', '<assistant>', '정의', '예시', '</s>', '<pad>']
    correct_probabilities = [0.2, 0.1, 0.6, 0.8, 0.5, 0.25, 0.99]
    mask = [0, 0, 0, 1, 1, 1, 0]
    terms = [-math.log(p) for p, include in zip(correct_probabilities, mask) if include]
    loss = sum(terms) / len(terms)
    response_probability = math.prod([p for p, include in zip(correct_probabilities, mask) if include])
    assert math.isclose(math.exp(-sum(terms)), response_probability)
    return {'tokens': tokens, 'correct_token_probabilities': correct_probabilities,
            'response_loss_mask': mask, 'response_loss': loss,
            'response_probability': response_probability, 'perplexity': math.exp(loss),
            'unmasked_loss': sum(-math.log(p) for p in correct_probabilities)/len(tokens)}


def reward_demo():
    # 같은 질문에 대한 점수의 절댓값보다 점수 차이를 비교한다.
    pairs = [(0, 0), (2, 0), (0, 2), (102, 100)]
    rows = [{'chosen_score': a, 'rejected_score': b, 'difference': a-b,
             'preference_probability': sigmoid(a-b), 'loss': softplus(-(a-b))}
            for a, b in pairs]
    assert rows[1]['loss'] == rows[3]['loss'], '양쪽 점수의 공통 이동은 차이를 바꾸지 않음'
    return rows


def dpo_terms(pc, pr, rc, rr, beta):
    delta = (pc - rc) - (pr - rr)
    return {'delta': delta, 'scaled_margin': beta*delta,
            'preference_probability': sigmoid(beta*delta), 'loss': softplus(-beta*delta)}


def dpo_demo(beta, steps, lr):
    example = dpo_terms(-3.0, -5.5, -4.0, -5.0, beta)
    ref_chosen = 0.35
    theta_ref = logit(ref_chosen)
    theta = theta_ref
    history = []
    def objective(t):
        return softplus(-beta*(t-theta_ref))
    for step in range(steps+1):
        p = sigmoid(theta)
        terms = dpo_terms(math.log(p), math.log1p(-p), math.log(ref_chosen),
                          math.log1p(-ref_chosen), beta)
        history.append({'step': step, 'p_chosen': p, 'loss': terms['loss'], 'delta': terms['delta']})
        if step < steps:
            gradient = beta * (sigmoid(beta*(theta-theta_ref)) - 1.0)
            if step == 0:
                eps = 1e-5
                numeric = (objective(theta+eps)-objective(theta-eps))/(2*eps)
                assert math.isclose(gradient, numeric, abs_tol=1e-8), 'DPO 기울기 유한차분 검산'
            theta -= lr * gradient
    assert math.isclose(history[0]['loss'], math.log(2)), '기준과 같은 모델의 DPO 초기 손실'
    assert history[-1]['loss'] < history[0]['loss'], '이 단일 쌍의 경사하강은 손실을 줄여야 함'
    return {'beta': beta, 'learning_rate': lr, 'reference_chosen_probability': ref_chosen,
            'fixed_log_probability_example': example, 'history': history,
            'note': '질문 하나와 가능한 응답 두 개뿐인 정책. 실제 언어 모델·일반화 성능이 아님.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rank', type=int, default=2, choices=range(1,5), metavar='1..4')
    parser.add_argument('--bits', type=int, default=4, choices=range(2,9), metavar='2..8')
    parser.add_argument('--beta', type=float, default=0.1)
    parser.add_argument('--steps', type=int, default=40)
    parser.add_argument('--lr', type=float, default=0.5)
    parser.add_argument('--json', action='store_true', help='모든 수치와 DPO 이력을 JSON으로 출력')
    args = parser.parse_args()
    if not (0.01 <= args.beta <= 2 and 1 <= args.steps <= 1000 and 0 < args.lr <= 1):
        parser.error('0.01 <= beta <= 2, 1 <= steps <= 1000, 0 < lr <= 1 범위를 사용하세요.')
    data = {'scope': 'LLM을 훈련하지 않는 표준 라이브러리 CPU 개념 실습',
            'lora': lora_demo(args.rank), 'quantization': quantization_demo(args.bits),
            'sft': sft_demo(), 'reward': reward_demo(),
            'dpo': dpo_demo(args.beta, args.steps, args.lr)}
    if args.json:
        print(json.dumps(data, ensure_ascii=False, indent=2))
        return
    print('12장 CPU 개념 실습: 실제 LLM/GPU/API를 사용하지 않습니다.\n')
    l = data['lora']
    print('[1] LoRA: W0는 그대로 두고 변화량 BA만 저랭크로 표현')
    print(f"4×6 전체={l['full_parameters']}, rank {args.rank} LoRA={l['lora_parameters']}, alpha/r={l['scale']:.3f}")
    print('병합 출력:', ', '.join(f'{v:.6f}' for v in l['merged_output']))
    print(f"두 경로와 병합의 최대 오차={l['max_error']:.3e}")
    print(f"12288×12288 전체={l['large_example_full']:,}; rank 8 변경분={l['large_example_rank8']:,}\n")
    q = data['quantization']
    print('[2] 균일 양자화: NF4 재현이 아닌 블록 스케일의 개념 실습')
    print(f"{args.bits}비트={q['levels']}개 코드, 원래 값={q['values']}")
    print('전역 복원:', ', '.join(f'{v:.4f}' for v in q['global_reconstructed']))
    print('블록 복원:', ', '.join(f'{v:.4f}' for v in q['block_reconstructed']))
    print(f"전역 MSE={q['global_mse']:.6f}, 블록 MSE={q['block_mse']:.6f}\n")
    s = data['sft']
    print('[3] SFT: 프롬프트와 패딩을 손실 평균에서 제외')
    print('마스크:', s['response_loss_mask'])
    print(f"응답 확률={s['response_probability']:.6f}, 평균 NLL={s['response_loss']:.6f}, PPL={s['perplexity']:.6f}")
    print(f"마스크 없는 평균 NLL={s['unmasked_loss']:.6f} (다른 평가 대상)\n")
    print('[4] 쌍별 보상: 점수의 절댓값보다 차이가 중요')
    for row in data['reward']:
        print(f"r+={row['chosen_score']:>3}, r-={row['rejected_score']:>3}: 선호확률={row['preference_probability']:.6f}, 손실={row['loss']:.6f}")
    d = data['dpo']
    e = d['fixed_log_probability_example']
    print('\n[5] DPO: 기준 모델 대비 chosen/rejected 로그확률 비율의 변화')
    print(f"고정 예제 delta={e['delta']:.3f}, beta={args.beta}, 손실={e['loss']:.6f}")
    for row in d['history']:
        if row['step'] in {0,1,10,args.steps}:
            print(f"step={row['step']:>3}: p(chosen)={row['p_chosen']:.6f}, delta={row['delta']:.6f}, loss={row['loss']:.6f}")
    print('\n검산 통과: LoRA 병합 등가성, 로그확률의 합/확률의 곱, 보상 점수 이동 불변성, DPO 유한차분 기울기.')
    print('실제 모델의 품질·안전성·벤치마크 성능을 측정한 결과가 아닙니다.')


if __name__ == '__main__':
    main()

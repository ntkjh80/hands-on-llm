#!/usr/bin/env python3
"""3장 개념 실습: 표준 라이브러리만 사용하는 작은 causal Transformer.

학습되지 않은 무작위 가중치로 원리를 계산한다. 언어 능력/품질/속도 벤치마크가 아니다.
실행: python3 lab.py --help
"""
import argparse
import math
import random


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def matvec(x, w):
    """행벡터 x @ W. W의 각 행이 입력 차원에 대응한다."""
    return [sum(x[i] * w[i][j] for i in range(len(x))) for j in range(len(w[0]))]


def softmax(scores, temperature=1.0):
    if temperature <= 0:
        raise ValueError('temperature는 0보다 커야 합니다. greedy는 별도 argmax입니다.')
    scaled = [s / temperature for s in scores]
    pivot = max(scaled)
    values = [math.exp(s - pivot) for s in scaled]
    total = sum(values)
    return [v / total for v in values]


def weighted_sum(weights, values):
    return [sum(a * v[j] for a, v in zip(weights, values)) for j in range(len(values[0]))]


def rmsnorm(x, eps=1e-6):
    # 교육용으로 학습 가능한 scale gamma는 모두 1로 둔다.
    scale = math.sqrt(sum(v*v for v in x) / len(x) + eps)
    return [v / scale for v in x]


def rope(x, position, base=10000.0):
    """인접 두 좌표씩 회전한다. 모델 구현에 따라 좌표 짝짓기 배치는 다를 수 있다."""
    out = []
    for pair in range(len(x) // 2):
        angle = position / (base ** (2 * pair / len(x)))
        a, b = x[2*pair:2*pair+2]
        c, s = math.cos(angle), math.sin(angle)
        out.extend([a*c-b*s, a*s+b*c])
    return out


def add(a, b):
    return [x + y for x, y in zip(a, b)]


class TinyTransformer:
    """2층, 2 heads, d_model=4, head_dim=2, vocab=8의 causal 모델.

    Pre-RMSNorm -> MHA+RoPE -> residual -> Pre-RMSNorm -> ReLU FFN -> residual.
    현대 구조 요소를 섞은 교육용 모델이며 Phi-3 또는 Llama의 재현이 아니다.
    """
    def __init__(self, seed=7):
        self.d, self.heads, self.dh, self.layers, self.vocab = 4, 2, 2, 2, 8
        rng = random.Random(seed)
        def matrix(rows, cols):
            return [[rng.uniform(-0.4, 0.4) for _ in range(cols)] for _ in range(rows)]
        self.emb = matrix(self.vocab, self.d)
        self.blocks = [dict(q=matrix(4, 4), k=matrix(4, 4), v=matrix(4, 4), o=matrix(4, 4),
                            up=matrix(4, 8), down=matrix(8, 4)) for _ in range(self.layers)]
        self.lm_head = matrix(self.d, self.vocab)
        self.projected_positions = 0

    def new_cache(self):
        return [[{'k': [], 'v': []} for _ in range(self.heads)] for _ in range(self.layers)]

    def projections(self, x, position, block):
        self.projected_positions += 1
        normed = rmsnorm(x)
        q, k, v = [matvec(normed, block[name]) for name in ('q', 'k', 'v')]
        return [(rope(q[h*2:h*2+2], position), rope(k[h*2:h*2+2], position), v[h*2:h*2+2])
                for h in range(self.heads)]

    def attend(self, q, keys, values):
        scores = [dot(q, k) / math.sqrt(self.dh) for k in keys]
        return weighted_sum(softmax(scores), values)

    def finish_block(self, residual, head_outputs, block):
        concatenated = [value for head in head_outputs for value in head]
        x = add(residual, matvec(concatenated, block['o']))
        expanded = matvec(rmsnorm(x), block['up'])
        return add(x, matvec([max(0, value) for value in expanded], block['down']))

    def full(self, tokens):
        xs = [self.emb[token][:] for token in tokens]
        for block in self.blocks:
            projected = [self.projections(x, pos, block) for pos, x in enumerate(xs)]
            ys = []
            for pos, residual in enumerate(xs):
                heads = []
                for h in range(self.heads):
                    q = projected[pos][h][0]
                    # causal mask의 구현: 자기 자신과 이전 위치만 선택한다.
                    keys = [projected[j][h][1] for j in range(pos+1)]
                    values = [projected[j][h][2] for j in range(pos+1)]
                    heads.append(self.attend(q, keys, values))
                ys.append(self.finish_block(residual, heads, block))
            xs = ys
        return [matvec(rmsnorm(x), self.lm_head) for x in xs]

    def step(self, token, position, cache):
        x = self.emb[token][:]
        for layer, block in enumerate(self.blocks):
            projected = self.projections(x, position, block)
            heads = []
            for h, (q, k, v) in enumerate(projected):
                cache[layer][h]['k'].append(k)
                cache[layer][h]['v'].append(v)
                heads.append(self.attend(q, cache[layer][h]['k'], cache[layer][h]['v']))
            x = self.finish_block(x, heads, block)
        return matvec(rmsnorm(x), self.lm_head)


def argmax(values):
    return max(range(len(values)), key=values.__getitem__)


def kv_bytes(batch, layers, length, kv_heads, head_dim, bytes_per_element):
    return 2 * batch * layers * length * kv_heads * head_dim * bytes_per_element


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tokens', type=int, default=8, help='비교할 생성 토큰 수 (1~64)')
    parser.add_argument('--temperature', type=float, default=0.8, help='샘플링 시연의 양수 온도')
    parser.add_argument('--seed', type=int, default=7, help='고정 난수 seed')
    args = parser.parse_args()
    if not 1 <= args.tokens <= 64:
        parser.error('--tokens는 1~64이어야 합니다.')
    if not math.isfinite(args.temperature) or args.temperature <= 0:
        parser.error('--temperature는 유한한 양수여야 합니다.')

    print('3장 실습 | 무학습 교육용 계산, CPU, 표준 라이브러리')
    print('\n[1] 한 attention head의 점수와 가중합')
    q = [1.0, 0.0]
    keys = [[0.0, 1.0], [2.0, 0.0], [1.0, 1.0]]
    values = [[1.0, 0.0], [0.0, 2.0], [2.0, 1.0]]
    scores = [dot(q, key) / math.sqrt(2) for key in keys]
    weights = softmax(scores)
    out = weighted_sum(weights, values)
    print('scaled scores:', [round(v, 6) for v in scores])
    print('attention weights:', [round(v, 6) for v in weights], 'sum=', round(sum(weights), 6))
    print('weighted value:', [round(v, 6) for v in out])
    assert abs(sum(weights)-1) < 1e-12

    print('\n[2] LM head logits -> 확률 -> 선택')
    logits = [2.0, 1.0, 0.0, -1.0]
    probs = softmax(logits, args.temperature)
    rng = random.Random(args.seed)
    draws = rng.choices(range(len(logits)), weights=probs, k=20)
    print('logits:', logits, 'temperature:', args.temperature)
    print('probabilities:', [round(p, 6) for p in probs])
    print('greedy id:', argmax(logits), '| sample ids (20 draws):', draws)

    print('\n[3] causal Transformer: 전체 재계산과 KV cache 비교')
    model = TinyTransformer(args.seed)
    tokens = [0, 1, 2]
    cache = model.new_cache()
    cached = None
    for pos, token in enumerate(tokens):
        cached = model.step(token, pos, cache)
    cached_count = model.projected_positions
    full_count = 0
    errors, generated = [], []
    for generation_step in range(args.tokens):
        before = model.projected_positions
        full = model.full(tokens)[-1]
        full_count += model.projected_positions - before
        error = max(abs(a-b) for a,b in zip(cached, full))
        errors.append(error)
        assert error < 1e-10, '캐시와 전체 계산의 logits가 다릅니다.'
        next_id = argmax(cached)
        generated.append(next_id)
        tokens.append(next_id)
        # 마지막 선택 이후에는 다음 분포가 필요 없으므로 계산하지 않는다.
        if generation_step < args.tokens-1:
            before = model.projected_positions
            cached = model.step(next_id, len(tokens)-1, cache)
            cached_count += model.projected_positions - before
    print('generated arbitrary token IDs:', generated)
    print('maximum logit difference: %.3g' % max(errors))
    print('QKV 투영 대상 위치 수 (층 수 포함): full=%d, cache=%d' % (full_count, cached_count))
    print('이는 계산 횟수 비교이며 실제 GPU 속도 배율이 아닙니다.')
    assert full_count == model.layers * sum(3+i for i in range(args.tokens))
    assert cached_count == model.layers * (3+args.tokens-1)
    short = model.full([0,1,2])
    extended = model.full([0,1,2,7])
    prefix_error = max(abs(a-b) for x,y in zip(short,extended[:3]) for a,b in zip(x,y))
    assert prefix_error < 1e-10, '미래 토큰이 이전 위치에 영향을 주었습니다.'
    print('future-token prefix invariance error: %.3g' % prefix_error)

    print('\n[4] KV cache의 이론 메모리 (모델 가중치/버퍼 제외)')
    for name, hkv in [('MHA', 32), ('GQA', 8), ('MQA', 1)]:
        size = kv_bytes(1,32,4096,hkv,128,2)
        print('%s: Hkv=%d, %.4f GiB' % (name,hkv,size/1024**3))

    print('\n[5] RoPE: 회전 전후 길이와 상대 위치')
    q, k, m, n = [1.0, 2.0], [3.0, -1.0], 2, 5
    left = dot(rope(q,m), rope(k,n))
    right = dot(q,rope(k,n-m))
    shifted = dot(rope(q,m+10),rope(k,n+10))
    print('norm² before/after:', dot(q,q), round(dot(rope(q,m),rope(q,m)),12))
    print('rotated dot = %.12f; relative dot = %.12f; shifted dot = %.12f' % (left,right,shifted))
    assert abs(left-right) < 1e-12 and abs(left-shifted) < 1e-12
    print('\nPASS: 확률합, 캐시 동치, causal 불변성, 메모리 계산, RoPE 상대 위치를 확인했습니다.')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""9장 멀티모달 원리: 표준 라이브러리만 사용하는 교육용 계산 실습.

실제 CLIP/BLIP-2 모델, 학습된 임베딩, 사진 인식 결과를 사용하지 않습니다.
작은 합성 이미지·명시적으로 만든 벡터로 연산과 차원을 검산합니다.
"""
import argparse
import math


def dot(a, b):
    if len(a) != len(b):
        raise ValueError("내적 벡터의 길이가 다릅니다.")
    return sum(x * y for x, y in zip(a, b))


def normalize(v):
    norm = math.sqrt(dot(v, v))
    if norm <= 0:
        raise ValueError("영벡터의 코사인 유사도는 정의할 수 없습니다.")
    return [x / norm for x in v]


def softmax(logits):
    if not logits:
        raise ValueError("softmax에는 하나 이상의 값이 필요합니다.")
    peak = max(logits)
    exps = [math.exp(x - peak) for x in logits]
    total = sum(exps)
    return [x / total for x in exps]


def logsumexp(values):
    peak = max(values)
    return peak + math.log(sum(math.exp(x - peak) for x in values))


def patchify(image, size):
    height, width = len(image), len(image[0])
    if size <= 0 or height % size or width % size:
        raise ValueError("이 실습에서는 영상의 높이와 너비가 패치 크기로 나누어떨어져야 합니다.")
    return [
        [channel for y in range(top, top + size)
         for x in range(left, left + size) for channel in image[y][x]]
        for top in range(0, height, size)
        for left in range(0, width, size)
    ]


def linear_project(rows, weights):
    if not rows or not weights or any(len(row) != len(weights) for row in rows):
        raise ValueError("입력과 투영 행렬의 차원이 맞지 않습니다.")
    if not weights[0] or any(len(row) != len(weights[0]) for row in weights):
        raise ValueError("투영 행렬은 직사각형이어야 합니다.")
    columns = list(zip(*weights))
    return [[dot(row, col) for col in columns] for row in rows]


def similarity_matrix(images, texts):
    vi = [normalize(v) for v in images]
    vt = [normalize(v) for v in texts]
    return [[dot(v, t) for t in vt] for v in vi]


def symmetric_contrastive_loss(scores, temperature):
    """행 i와 열 i가 정답 쌍인 정사각형 배치의 대칭 cross entropy."""
    if temperature <= 0:
        raise ValueError("온도는 0보다 커야 합니다.")
    n = len(scores)
    if not n or any(len(row) != n for row in scores):
        raise ValueError("이 대조 손실 예제는 같은 수의 이미지·캡션 쌍이 필요합니다.")
    logits = [[x / temperature for x in row] for row in scores]
    row_losses = [logsumexp(row) - row[i] for i, row in enumerate(logits)]
    columns = list(zip(*logits))
    col_losses = [logsumexp(col) - col[i] for i, col in enumerate(columns)]
    return sum(row_losses + col_losses) / (2 * n)


def attention(queries, keys, values):
    """교육용 단일 head cross-attention. Q-Former 전체 구현이 아닙니다."""
    if len(keys) != len(values) or not keys or not values:
        raise ValueError("key와 value는 대응되는 동일 개수의 행이어야 합니다.")
    dk = len(keys[0])
    if not dk or any(len(k) != dk for k in keys) or any(len(q) != dk for q in queries):
        raise ValueError("query와 key의 차원이 맞지 않습니다.")
    dv = len(values[0])
    if any(len(v) != dv for v in values):
        raise ValueError("value의 폭이 일정해야 합니다.")
    weights = [softmax([dot(q, k) / math.sqrt(dk) for k in keys]) for q in queries]
    outputs = [[sum(wi * values[i][j] for i, wi in enumerate(w))
                for j in range(dv)] for w in weights]
    return weights, outputs


def show_matrix(rows, row_names, col_names):
    print(f"{'':>12}" + ''.join(f'{name:>12}' for name in col_names))
    for name, row in zip(row_names, rows):
        print(f'{name:>12}' + ''.join(f'{x:12.4f}' for x in row))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--patch-size', type=int, choices=[1, 2, 4, 8], default=2,
                        help='8×8 합성 RGB 이미지의 패치 한 변 (기본 2)')
    parser.add_argument('--temperature', type=float, default=0.2,
                        help='대조 손실의 양수 온도 (기본 0.2)')
    parser.add_argument('--query-scale', type=float, default=2.0,
                        help='query 벡터 배율, 0이면 균등 attention (기본 2)')
    args = parser.parse_args()
    if not math.isfinite(args.temperature) or args.temperature <= 0:
        parser.error('--temperature는 0보다 큰 유한한 수여야 합니다.')
    if not math.isfinite(args.query_scale):
        parser.error('--query-scale은 유한한 수여야 합니다.')

    print('9장 멀티모달 원리 실습 — 합성 데이터, 실제 모델 추론 아님')
    print('난수를 사용하지 않는 결정적 계산입니다. 사진·모델·외부 패키지 다운로드 없음.\n')

    print('1. 패치 분할과 선형 투영')
    image = [[[y / 7, x / 7, (x + y) / 14] for x in range(8)] for y in range(8)]
    patches = patchify(image, args.patch_size)
    # 세 출력은 각 패치 RGB 채널 평균. 학습된 ViT 가중치가 아닌 해석 가능한 예제.
    pixels_per_patch = args.patch_size ** 2
    weights = [[(1 / pixels_per_patch if i % 3 == j else 0) for j in range(3)]
               for i in range(pixels_per_patch * 3)]
    features = linear_project(patches, weights)
    print(f'입력: 1×3×8×8, 패치 한 변: {args.patch_size}')
    print(f'패치 행렬: {len(patches)}×{len(patches[0])}, 투영 출력: {len(features)}×3')
    print('첫 패치의 채널 평균:', [round(x, 6) for x in features[0]])
    print(f'클래스 토큰 포함 attention 점수 개수: {(len(patches) + 1) ** 2}')
    assert len(patches) * len(patches[0]) == 8 * 8 * 3, '패치 분할 중 값이 손실되었습니다.'
    expected_red = (args.patch_size - 1) / 14
    assert math.isclose(features[0][0], expected_red, abs_tol=1e-12), '평균 투영 검산 실패'

    print('\n2. 이미지·문장 공동 공간의 코사인 유사도')
    print('cat/car/tree는 벡터에 붙인 교육용 이름입니다. 인식 모델이 추출한 벡터가 아닙니다.')
    images = [[3, 0.5, 0], [0.2, 2, 0.2], [0, 0.3, 2.5]]
    texts = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    names = ['cat', 'car', 'tree']
    scores = similarity_matrix(images, texts)
    show_matrix(scores, names, names)
    predictions = [max(range(3), key=lambda j: row[j]) for row in scores]
    print('각 이미지에서 선택한 문장:', [names[i] for i in predictions])
    assert predictions == [0, 1, 2], '의도한 합성 대응과 다른 순위입니다.'
    scaled = similarity_matrix([[x * 10 for x in v] for v in images], texts)
    assert all(math.isclose(a, b, abs_tol=1e-12) for row_a, row_b in zip(scores, scaled)
               for a, b in zip(row_a, row_b)), '코사인의 크기 불변성 검산 실패'

    print('\n3. 온도와 대조 손실')
    print('실제 학습을 수행하지 않고, 고정된 점수와 정답 대응의 손실을 계산합니다.')
    for temp in [1.0, args.temperature]:
        probs = softmax([s / temp for s in scores[0]])
        loss = symmetric_contrastive_loss(scores, temp)
        print(f'온도 {temp:g}: 첫 행 후보 가중치 {[round(p, 6) for p in probs]}, 대칭 손실 {loss:.6f}')
        assert math.isclose(sum(probs), 1.0, abs_tol=1e-12)
    wrong_pair_scores = [[row[1], row[0], row[2]] for row in scores]
    correct_loss = symmetric_contrastive_loss(scores, args.temperature)
    wrong_loss = symmetric_contrastive_loss(wrong_pair_scores, args.temperature)
    print(f'앞의 두 캡션만 뒤섞고 정답 인덱스를 그대로 둔 경우 손실: {wrong_loss:.6f}')
    assert wrong_loss > correct_loss, '이 합성 예제에서는 잘못된 대응이 더 큰 손실이어야 합니다.'
    uniform = [[0.5] * 3 for _ in range(3)]
    collapse = symmetric_contrastive_loss(uniform, args.temperature)
    print(f'모든 유사도가 동일한 경우 손실: {collapse:.6f} = log(3)')
    assert math.isclose(collapse, math.log(3), abs_tol=1e-10)

    print('\n4. query가 시각 특징을 읽는 작은 attention')
    keys = [[1, 0], [0, 1], [-1, 0], [0, -1]]
    values = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0]]
    queries = [[args.query_scale, 0], [0, args.query_scale]]
    attn, summaries = attention(queries, keys, values)
    show_matrix(attn, ['query-x', 'query-y'], ['patch-A', 'patch-B', 'patch-C', 'patch-D'])
    show_matrix(summaries, ['query-x', 'query-y'], ['feature-1', 'feature-2', 'feature-3'])
    print('query 수 2, key 수 4, value 폭 3 → attention 2×4, 요약 2×3')
    for row in attn:
        assert math.isclose(sum(row), 1.0, abs_tol=1e-12)
    uniform_attention, _ = attention([[0, 0]], keys, values)
    assert all(math.isclose(w, 0.25) for w in uniform_attention[0])

    print('\n5. LLM 입력 폭에 맞추는 투영')
    bridge = [[1, 0, 0.5, -0.5], [0, 1, 0.5, 0.5], [0.5, 0.5, 1, 0]]
    prompts = linear_project(summaries, bridge)
    show_matrix(prompts, ['prompt-1', 'prompt-2'], ['dim-1', 'dim-2', 'dim-3', 'dim-4'])
    print('2×3 요약 × 3×4 투영 → 2×4 연속 벡터. query 개수는 유지됩니다.')
    print('LLM을 연결하지 않았으므로 문장·캡션은 생성하지 않습니다.')
    assert len(prompts) == len(queries) and all(len(row) == 4 for row in prompts)
    print('\n검산 통과: 값 보존·평균·유사도 순위·크기 불변성·확률 합·대응 손실·균등 attention·투영 형상')


if __name__ == '__main__':
    main()

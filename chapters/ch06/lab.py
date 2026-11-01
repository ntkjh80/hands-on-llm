#!/usr/bin/env python3
"""6장 개념 실습: Python 표준 라이브러리, CPU, 네트워크·LLM 호출 없음.

교육용 로짓·문자 가중치·후보 답을 사용한다. 실제 모델 성능 측정이 아니다.
예: python3 lab.py --temperature 2 --top-p 0.8 --top-k 0 --seed 6
"""
import argparse
from collections import Counter
import json
import math
import random
import re

TOKENS = ('A', 'B', 'C', 'D')
LOGITS = (2.0, 1.0, 0.0, -1.0)
LABELS = ('positive', 'neutral', 'negative')
SOURCE = '중복 결제 확인 요청. 같은 상품이 두 번 결제되었습니다.'


def normalize(values):
    total = math.fsum(values)
    if not total > 0 or any(v < 0 or not math.isfinite(v) for v in values):
        raise ValueError('정규화할 확률 질량이 없거나 값이 잘못되었습니다.')
    return [v / total for v in values]


def softmax(logits, temperature):
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError('temperature는 유한한 양수여야 합니다.')
    maximum = max(logits)
    return normalize([math.exp((z - maximum) / temperature) for z in logits])


def truncate(probabilities, top_k=0, top_p=1.0):
    """이 교육용 구현의 순서: top-k → 재정규화 → top-p → 재정규화."""
    if top_k < 0 or not 0 < top_p <= 1:
        raise ValueError('top_k >= 0, 0 < top_p <= 1 조건이 필요합니다.')
    ranked = sorted(range(len(probabilities)), key=lambda i: (-probabilities[i], i))
    if top_k:
        ranked = ranked[:top_k]
    after_k = normalize([p if i in ranked else 0.0 for i, p in enumerate(probabilities)])
    kept = []
    cumulative = 0.0
    for index in ranked:
        if after_k[index] == 0:
            continue
        kept.append(index)
        cumulative += after_k[index]
        # 임계값을 넘게 하는 마지막 후보도 포함한다.
        if cumulative >= top_p - 1e-12:
            break
    return normalize([p if i in kept else 0.0 for i, p in enumerate(after_k)])


def entropy(probabilities):
    return -math.fsum(p * math.log2(p) for p in probabilities if p > 0)


def majority_probability(q, n):
    """독립·동일 정확도·이진 정오·홀수 표본이라는 가정의 계산."""
    if not 0 <= q <= 1 or n < 1 or n % 2 == 0:
        raise ValueError('q는 0..1, n은 양의 홀수여야 합니다.')
    return math.fsum(math.comb(n, k) * q**k * (1-q)**(n-k)
                     for k in range(n//2 + 1, n+1))


def normalize_answer(answer):
    match = re.fullmatch(r'\s*(\d+)\s*(?:자루)?\s*', answer)
    if not match:
        raise ValueError('이 실습은 음이 아닌 정수 및 선택적인 자루 단위만 허용합니다.')
    return int(match.group(1))


def allowed_next(prefix):
    matching = [label for label in LABELS if label.startswith(prefix)]
    chars = sorted({label[len(prefix)] for label in matching if len(label) > len(prefix)})
    return chars, prefix in LABELS


def constrained_label(rng):
    """문자 단위 trie. 실제 LLM 토크나이저나 JSON 문법 구현은 아니다."""
    prefix = ''
    trace = []
    for _ in range(20):
        allowed, can_end = allowed_next(prefix)
        choices = allowed + (['<END>'] if can_end else [])
        if not choices:
            raise ValueError('허용되는 다음 문자가 없습니다.')
        # 교육용 임의 가중치. 실제 모델이 내놓은 점수가 아니다.
        weights = [1.0 if c == '<END>' else 1.0 + ord(c) % 5 for c in choices]
        selected = rng.choices(choices, weights=weights, k=1)[0]
        trace.append((prefix or '∅', ','.join(choices), selected))
        if selected == '<END>':
            return prefix, trace
        prefix += selected
    raise RuntimeError('종료 전에 길이 제한에 도달했습니다.')


def reject_nonstandard_number(value):
    raise ValueError('표준 JSON이 아닌 숫자 상수: ' + value)


def validate_ticket(text):
    """파싱 → 스키마 → 원문 내 근거 포함 여부. 사실성 전체를 보장하지 않는다."""
    try:
        obj = json.loads(text, parse_constant=reject_nonstandard_number)
    except (json.JSONDecodeError, ValueError) as error:
        return '파싱 실패', type(error).__name__
    if not isinstance(obj, dict) or set(obj) != {'label', 'priority', 'evidence'}:
        return '스키마 실패', '객체와 정확한 필수 키 세 개가 필요함'
    if obj['label'] not in ('billing', 'delivery', 'product'):
        return '스키마 실패', '허용되지 않은 label'
    # bool은 Python에서 int의 하위형이므로 isinstance(..., int)만 쓰지 않는다.
    if type(obj['priority']) is not int or not 1 <= obj['priority'] <= 3:
        return '스키마 실패', 'priority는 1..3의 정수여야 함'
    if not isinstance(obj['evidence'], str) or not obj['evidence'].strip():
        return '스키마 실패', 'evidence는 비어 있지 않은 문자열이어야 함'
    if obj['evidence'] not in SOURCE:
        return '근거 검사 실패', 'evidence가 입력의 직접 인용 구절이 아님'
    return '정의한 검사 통과', '파싱·스키마·근거 포함 검사만 통과'


def beam_search(width, target=10, depth=3):
    """상태 1에서 +3 또는 ×2. 거리 휴리스틱으로 상태를 남긴다."""
    frontier = [(1, '1')]
    trace = []
    for step in range(1, depth+1):
        candidates = {}
        for value, path in frontier:
            for next_value, op in ((value+3, '+3'), (value*2, '×2')):
                candidates.setdefault(next_value, (next_value, f'{path} {op} → {next_value}'))
        ranked = sorted(candidates.values(), key=lambda item: (abs(item[0]-target), item[0]))
        frontier = ranked[:width]
        trace.append((step, [v for v, _ in ranked], [v for v, _ in frontier]))
        for value, path in frontier:
            if value == target:
                return True, path, trace
    return False, frontier[0][1], trace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--top-p', type=float, default=0.8)
    parser.add_argument('--top-k', type=int, default=0, help='0이면 개수 제한 없음')
    parser.add_argument('--seed', type=int, default=6)
    parser.add_argument('--draws', type=int, default=2000)
    parser.add_argument('--beam-width', type=int, default=2)
    args = parser.parse_args()
    if not math.isfinite(args.temperature) or args.temperature <= 0:
        parser.error('--temperature는 유한한 양수여야 합니다.')
    if not 0 < args.top_p <= 1 or args.top_k < 0:
        parser.error('0 < --top-p <= 1 및 --top-k >= 0이 필요합니다.')
    if args.draws < 1 or args.beam_width < 1:
        parser.error('--draws와 --beam-width는 양수여야 합니다.')
    rng = random.Random(args.seed)
    print('6장 프롬프트 엔지니어링 개념 실습')
    print('교육용 가상 데이터. LLM/API/GPU 사용 없음. 실제 모델 성능이 아닙니다.')
    print(f'seed={args.seed}; 샘플 수={args.draws}')
    print('\n[1] 온도와 후보 제한')
    for t in (0.5, 1.0, 2.0):
        probs = softmax(LOGITS, t)
        print(f'T={t:.1f}: ' + ', '.join(f'{s}={p:.6f}' for s,p in zip(TOKENS, probs)))
    before = softmax(LOGITS, args.temperature)
    after = truncate(before, args.top_k, args.top_p)
    assert math.isclose(sum(after), 1.0, abs_tol=1e-12), '확률 보존 검산'
    print(f'선택 설정: T={args.temperature}, top_k={args.top_k}, top_p={args.top_p}')
    print('최종 분포: ' + ', '.join(f'{s}={p:.6f}' for s,p in zip(TOKENS, after)))
    print(f'엔트로피: {entropy(before):.6f} → {entropy(after):.6f} bits')
    counts = Counter(rng.choices(TOKENS, weights=after, k=args.draws))
    print('실제 표본 빈도: ' + ', '.join(f'{s}={counts[s]}/{args.draws}' for s in TOKENS))
    print('\n[2] 최종 답 정규화와 투표')
    answers = ['21', '21자루', '20', '21', '22']
    votes = Counter(normalize_answer(a) for a in answers)
    print('교사가 만든 후보 답:', answers)
    print('정규화한 표:', dict(sorted(votes.items())))
    winner, count = votes.most_common(1)[0]
    print(f'다수 답={winner}, 표={count}/{len(answers)}; 실제 정답은 별도 검산해야 함')
    for n in (1, 3, 5, 9):
        print(f'가정 q=0.7, n={n}: 독립={majority_probability(0.7,n):.6f}, 완전 상관=0.700000')
    print('\n[3] 중간 상태를 남기는 탐색')
    for width in sorted({1, args.beam_width}):
        found, path, trace = beam_search(width)
        print(f'beam width={width}, 목표=10, 최대 깊이=3')
        for step, candidates, retained in trace:
            print(f'  깊이 {step}: 후보={candidates}; 유지={retained}')
        print(f'  목표 발견={found}; 경로={path}')
    print('\n[4] 문자 단위 문법 제약')
    for prefix in ('', 'n', 'ne', 'neg', 'negative', 'banana'):
        allowed, can_end = allowed_next(prefix)
        print(f'prefix={prefix!r}: 다음 문자={allowed}, 종료 허용={can_end}')
    generated, trace = constrained_label(random.Random(args.seed))
    print('가상 문자 점수로 생성한 라벨:', generated)
    for prefix, allowed, selected in trace:
        print(f'  {prefix:10} → 허용 {{{allowed}}} → {selected}')
    assert generated in LABELS, '허용 언어에 속하는지 검산'
    print('\n[5] JSON 파싱·스키마·근거 검사')
    cases = {
        '정상 예': '{"label":"billing","priority":2,"evidence":"중복 결제 확인 요청"}',
        '잘린 출력': '{"label":"billing",',
        '잘못된 자료형': '{"label":"billing","priority":"2","evidence":"중복 결제 확인 요청"}',
        '허용값 위반': '{"label":"other","priority":2,"evidence":"중복 결제 확인 요청"}',
        '자료에 없는 근거': '{"label":"billing","priority":2,"evidence":"환불 완료"}',
        'bool 함정': '{"label":"billing","priority":true,"evidence":"중복 결제 확인 요청"}',
    }
    for name, raw in cases.items():
        status, reason = validate_ticket(raw)
        print(f'{name}: {status} / {reason}')
    print('\n[6] 프롬프트 모듈 조립 — 모델에는 전송하지 않음')
    modules = {
        '작업': '문의 종류를 분류한다.',
        '범주': 'billing=결제, delivery=배송, product=제품 기능.',
        '형식': 'label, priority, evidence 세 필드의 JSON 객체만 출력한다.',
        '자료': SOURCE,
    }
    print('\n'.join(f'{name}: {value}' for name, value in modules.items()))
    print('\n검산 완료: 확률 합 및 문법 라벨. 이 결과는 실제 LLM 응답이 아닙니다.')


if __name__ == '__main__':
    main()

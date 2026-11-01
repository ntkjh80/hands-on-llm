#!/usr/bin/env python3
"""1장 개념 실습: BoW, 어텐션 가중합, 빈도 기반 자기회귀 생성.
표준 라이브러리만 사용한다. 신경망/LLM/API를 실행하지 않는다.
"""
import argparse
from collections import Counter, defaultdict
import math
import random

CORPUS = [
    '오늘 날씨가 좋다', '오늘 날씨가 좋다', '오늘 날씨가 흐리다',
    '오늘 기분이 좋다', '내일 날씨가 춥다', '내일 기분이 좋다',
]
EOS = '<끝>'


def tokenize(text):
    """교육용 공백 분할. 실제 LLM 토크나이저가 아니다."""
    return text.split()


def bow(text, vocabulary):
    counts = Counter(tokenize(text))
    return [counts[word] for word in vocabulary]


def cosine(left, right):
    denom = math.sqrt(sum(x*x for x in left)*sum(x*x for x in right))
    if denom == 0:
        raise ValueError('영벡터의 코사인 유사도는 정의되지 않습니다.')
    return sum(x*y for x, y in zip(left, right))/denom


def softmax(scores):
    shift = max(scores)
    exp_scores = [math.exp(s-shift) for s in scores]
    total = sum(exp_scores)
    return [x/total for x in exp_scores]


def fit_bigram(corpus):
    """신경망 학습이 아닌, 바로 앞 토큰별 다음 토큰의 실제 횟수 집계."""
    counts = defaultdict(Counter)
    for sentence in corpus:
        tokens = tokenize(sentence) + [EOS]
        for previous, following in zip(tokens, tokens[1:]):
            counts[previous][following] += 1
    return counts


def next_probabilities(counts, previous):
    if previous not in counts:
        raise ValueError(f'학습 문장에 다음 토큰 통계가 없는 입력: {previous!r}')
    total = sum(counts[previous].values())
    return {token: count/total for token, count in sorted(counts[previous].items())}


def generate(counts, prompt, max_new, context_window, sample, seed):
    tokens = tokenize(prompt)
    if not tokens:
        raise ValueError('프롬프트에 최소 한 토큰이 필요합니다.')
    if len(tokens) >= context_window:
        raise ValueError('입력 토큰 수는 문맥 창보다 작아야 합니다.')
    rng = random.Random(seed)
    trace = []
    for _ in range(min(max_new, context_window-len(tokens))):
        probs = next_probabilities(counts, tokens[-1])
        candidates = list(probs)
        if sample:
            chosen = rng.choices(candidates, weights=list(probs.values()), k=1)[0]
        else:
            chosen = max(candidates, key=lambda token: probs[token])
        trace.append((tokens[-1], probs, chosen))
        if chosen == EOS:
            break
        tokens.append(chosen)
    return tokens, trace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=7, help='표본 선택용 고정 난수 시드')
    parser.add_argument('--sample', action='store_true', help='최빈값 대신 분포에서 표본 선택')
    parser.add_argument('--prompt', default='오늘', help='공백 토큰화 입력. 기본값: 오늘')
    parser.add_argument('--max-new-tokens', type=int, default=6)
    parser.add_argument('--context-window', type=int, default=8)
    parser.add_argument('--attention-score', type=float, default=3.0,
                        help='세 번째 입력의 교육용 어텐션 점수; 앞 점수는 1, 2')
    args = parser.parse_args()
    if args.max_new_tokens < 1 or args.context_window < 2:
        parser.error('max-new-tokens는 1 이상, context-window는 2 이상이어야 합니다.')
    if not math.isfinite(args.attention_score):
        parser.error('attention-score는 유한한 숫자여야 합니다.')
    print('1장 개념 실습 — LLM 실행이 아닌 작은 계산 모형')
    print('\n[1] Bag-of-Words: 순서를 바꿔도 빈도는 같다')
    a, b, c = '고양이 가 강아지 를 쫓는다', '강아지 가 고양이 를 쫓는다', '고양이 고양이 잠잔다'
    vocab = list(dict.fromkeys(tokenize(a)+tokenize(b)+tokenize(c)))
    vectors = [bow(doc, vocab) for doc in (a,b,c)]
    print('어휘:', vocab)
    for label, doc, vector in zip('ABC', (a,b,c), vectors):
        print(f'{label}: {doc} -> {vector}')
    print(f'cos(A, B) = {cosine(vectors[0], vectors[1]):.6f}')
    print(f'cos(A, C) = {cosine(vectors[0], vectors[2]):.6f}')
    assert vectors[0] == vectors[1], '단어 순열은 단어 빈도를 보존해야 합니다.'

    print('\n[2] 어텐션: 가중치 합은 1, 문맥 벡터는 가중합')
    states = [(1.0,0.0), (0.0,1.0), (1.0,1.0)]
    scores = [1.0,2.0,args.attention_score]
    weights = softmax(scores)
    context = [sum(w*h[axis] for w,h in zip(weights, states)) for axis in range(2)]
    print('교육용 상태:', states)
    print('교육용 점수:', scores)
    print('softmax 가중치:', [round(w,6) for w in weights])
    print('가중합 문맥 벡터:', [round(x,6) for x in context])
    assert math.isclose(sum(weights), 1.0, abs_tol=1e-12)
    assert all(0 <= x <= 1 for x in context), '이 예제의 가중합은 입력의 볼록 범위 안에 있어야 합니다.'

    print('\n[3] 빈도 기반 bigram 언어 모형: 선택 -> 추가 -> 다시 예측')
    print('학습 문장:', CORPUS)
    print('이 모형은 바로 앞 토큰 하나만 기억하며, 의미를 이해하는 신경망이 아닙니다.')
    print(f'설정: prompt={args.prompt!r}, max_new={args.max_new_tokens}, context={args.context_window}, sample={args.sample}, seed={args.seed}')
    counts = fit_bigram(CORPUS)
    try:
        output, trace = generate(counts, args.prompt, args.max_new_tokens,
                                 args.context_window, args.sample, args.seed)
    except ValueError as error:
        parser.error(str(error))
    for step, (previous, distribution, selected) in enumerate(trace,1):
        formatted = ', '.join(f'{token}:{value:.3f}' for token,value in distribution.items())
        print(f'단계 {step} | 이전={previous} | {formatted} | 선택={selected}')
        assert math.isclose(sum(distribution.values()), 1.0, abs_tol=1e-12)
    print('최종 문자열:', ' '.join(output))
    print('종료 이유:', '종료 토큰' if trace and trace[-1][2] == EOS else '길이 제한')
    assert len(output) <= args.context_window
    print('\n검산 통과. 출력은 위 작은 말뭉치의 통계이며, 실제 LLM 성능이 아닙니다.')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""2장 CPU 실습: 작은 BPE, 풀링, skip-gram/negative-sampling 학습.

표준 라이브러리만 사용한다. 실제 LLM이나 원문의 음악 데이터가 아니다.
내장한 가상 재생목록을 이용하며 임베딩과 손실은 실제로 계산한다.
"""
import argparse
from collections import Counter
import math
import random

END = "</w>"


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def cosine(a, b):
    denom = math.sqrt(dot(a, a) * dot(b, b))
    return dot(a, b) / denom if denom else 0.0


def sigmoid(s):
    if s >= 0:
        return 1.0 / (1.0 + math.exp(-s))
    e = math.exp(s)
    return e / (1.0 + e)


def loss_for_score(s, y):
    # BCE(logits) = softplus(s) - y*s; 수치적으로 안정적인 계산.
    return max(s, 0.0) - y * s + math.log1p(math.exp(-abs(s)))


def merge_symbols(symbols, pair):
    result, i = [], 0
    while i < len(symbols):
        if i + 1 < len(symbols) and (symbols[i], symbols[i + 1]) == pair:
            result.append(symbols[i] + symbols[i + 1])
            i += 2
        else:
            result.append(symbols[i])
            i += 1
    return tuple(result)


def learn_bpe(word_counts, rounds):
    words = {tuple(word) + (END,): count for word, count in word_counts.items()}
    merges = []
    for _ in range(rounds):
        pairs = Counter()
        for symbols, count in words.items():
            for pair in zip(symbols, symbols[1:]):
                pairs[pair] += count
        if not pairs:
            break
        # 빈도 내림차순, 동률은 문자열 순서: 재현 가능한 교육용 정책.
        pair, count = sorted(pairs.items(), key=lambda item: (-item[1], item[0]))[0]
        merges.append((pair, count))
        merged = Counter()
        for symbols, frequency in words.items():
            merged[merge_symbols(symbols, pair)] += frequency
        words = dict(merged)
    return merges


def encode_word(word, merges):
    symbols = tuple(word) + (END,)
    for pair, _ in merges:
        symbols = merge_symbols(symbols, pair)
    return list(symbols)


def masked_mean(vectors, mask):
    if not vectors or len(vectors) != len(mask):
        raise ValueError("벡터와 마스크의 길이를 확인하세요.")
    if len({len(v) for v in vectors}) != 1:
        raise ValueError("모든 벡터의 차원이 같아야 합니다.")
    if not all(m in (0, 1) for m in mask) or sum(mask) == 0:
        raise ValueError("마스크는 0/1이며 적어도 한 위치가 포함되어야 합니다.")
    return [sum(v[j] * m for v, m in zip(vectors, mask)) / sum(mask)
            for j in range(len(vectors[0]))]


def update_pair(u, v, y, learning_rate):
    """업데이트 전 두 벡터를 사용해 동시 경사 하강을 한다."""
    old_u, old_v = u[:], v[:]
    score = dot(old_u, old_v)
    gradient = sigmoid(score) - y
    for j in range(len(u)):
        u[j] -= learning_rate * gradient * old_v[j]
        v[j] -= learning_rate * gradient * old_u[j]
    return loss_for_score(score, y)


SONGS = {
    "t01": ("햇살 산책", "가상 팝"), "t02": ("여름 편지", "가상 팝"),
    "t03": ("도시의 오후", "가상 팝"), "t04": ("반짝이는 길", "가상 팝"),
    "t05": ("깊은 기타", "가상 록"), "t06": ("밤의 드럼", "가상 록"),
    "t07": ("붉은 무대", "가상 록"), "t08": ("달리는 앰프", "가상 록"),
    "t09": ("고요한 피아노", "가상 앰비언트"), "t10": ("새벽 안개", "가상 앰비언트"),
    "t11": ("느린 파도", "가상 앰비언트"), "t12": ("별빛 호흡", "가상 앰비언트"),
}


def make_playlists(seed):
    rng = random.Random(seed)
    ids = list(SONGS)
    playlists = []
    for start in (0, 4, 8):
        group = ids[start:start + 4]
        for _ in range(8):
            playlist = group[:]
            rng.shuffle(playlist)
            playlists.append(playlist)
    return playlists


def positive_pairs(playlists, window):
    pairs = []
    for playlist in playlists:
        for i, center in enumerate(playlist):
            for j in range(max(0, i - window), min(len(playlist), i + window + 1)):
                if j != i:
                    pairs.append((center, playlist[j]))
    return pairs


def make_examples(positives, vocabulary, negatives, seed):
    rng = random.Random(seed)
    index = {word: i for i, word in enumerate(vocabulary)}
    result = []
    for center, context in positives:
        result.append((index[center], index[context], 1))
        # 교육용 설정: 전체 어휘에서 균등 추출. 중심과 현재 양성 항목만 제외한다.
        # 같은 그룹의 다른 곡이 음성이 될 수 있으며 false negative도 남는다.
        candidates = [w for w in vocabulary if w not in (center, context)]
        for _ in range(negatives):
            negative = rng.choice(candidates)
            result.append((index[center], index[negative], 0))
    return result


def average_loss(U, V, examples):
    return sum(loss_for_score(dot(U[i], V[j]), y) for i, j, y in examples) / len(examples)


def train_sgns(vocabulary, examples, dimensions, epochs, seed):
    rng = random.Random(seed)
    U = [[rng.uniform(-0.25, 0.25) for _ in range(dimensions)] for _ in vocabulary]
    V = [[rng.uniform(-0.25, 0.25) for _ in range(dimensions)] for _ in vocabulary]
    history = [(0, average_loss(U, V, examples))]
    order = list(range(len(examples)))
    checkpoints = {1, 10, epochs}
    for epoch in range(1, epochs + 1):
        rng.shuffle(order)
        rate = 0.08 * (1 - 0.75 * (epoch - 1) / max(epochs - 1, 1))
        for ix in order:
            i, j, y = examples[ix]
            update_pair(U[i], V[j], y, rate)
        if epoch in checkpoints:
            history.append((epoch, average_loss(U, V, examples)))
    return U, V, history


def run(args):
    print("2장 개념 실습 | 실제 LLM 없음 | 가상 노래 데이터 | 표준 Python CPU")
    print(f"seed={args.seed}, epochs={args.epochs}, dims={args.dims}, window={args.window}, negatives={args.negatives}")
    print("\n[1] 문자 수와 UTF-8 바이트 수")
    text = "A가😊"
    print(f"{text!r}: 코드 포인트={len(text)}, UTF-8 바이트={len(text.encode('utf-8'))}")
    print("UTF-8 bytes:", list(text.encode("utf-8")))
    assert text.encode("utf-8").decode("utf-8") == text

    print("\n[2] 작은 BPE: 실제 모델용 어휘가 아닌 단어별 병합 실험")
    words = {"low": 5, "lower": 2, "lowest": 2, "newer": 4, "wider": 3}
    merges = learn_bpe(words, 8)
    for step, (pair, count) in enumerate(merges, 1):
        print(f"병합 {step}: {pair[0]!r} + {pair[1]!r} -> {''.join(pair)!r} (빈도 {count})")
    for word in ("low", "lower", "lowest", "newer"):
        tokens = encode_word(word, merges)
        print(f"{word:6s} -> {tokens}")
        assert "".join(tokens).removesuffix(END) == word

    print("\n[3] 임베딩 lookup, 마스크 평균, 코사인")
    E = [[0., 0.], [2., 0.], [0., 2.], [10., 10.]]
    ids, mask = [1, 2, 3], [1, 1, 0]
    vectors = [E[i] for i in ids]
    pooled = masked_mean(vectors, mask)
    wrong = masked_mean(vectors, [1, 1, 1])
    print("ids:", ids, "lookup:", vectors)
    print("마스크 평균:", pooled, "무조건 평균:", wrong)
    print(f"cos([1,0],[1,1])={cosine([1,0],[1,1]):.6f}")
    assert pooled == [1., 1.] and wrong == [4., 4.]
    assert abs(cosine([1, 0], [1, 1]) - 1 / math.sqrt(2)) < 1e-12

    print("\n[4] 경사 업데이트 손계산 검산")
    u, v = [1., 0.], [0., 1.]
    before = sigmoid(dot(u, v))
    update_pair(u, v, 1, 0.1)
    after = sigmoid(dot(u, v))
    print(f"양성: u={u}, v={v}, sigmoid {before:.6f} -> {after:.6f}")
    assert u == [1., 0.05] and v == [0.05, 1.] and after > before
    u, v = [1., 0.], [0., 1.]
    update_pair(u, v, 0, 0.1)
    assert sigmoid(dot(u, v)) < 0.5

    print("\n[5] 가상 재생목록에서 skip-gram과 음성 샘플링")
    vocabulary = list(SONGS)
    playlists = make_playlists(args.seed)
    positives = positive_pairs(playlists, args.window)
    examples = make_examples(positives, vocabulary, args.negatives, args.seed + 1)
    print("첫 재생목록:", playlists[0])
    print("첫 양성 쌍:", positives[:6])
    print(f"재생목록={len(playlists)}, 곡={len(vocabulary)}, 양성 쌍={len(positives)}, 전체 학습 쌍={len(examples)}")
    print("음성은 균등 샘플링; 중심/현재 양성만 제외. false negative가 있을 수 있음.")
    U, V, history = train_sgns(vocabulary, examples, args.dims, args.epochs, args.seed + 2)
    for epoch, value in history:
        print(f"epoch={epoch:3d}, 고정 학습 쌍 평균 BCE={value:.6f}")
    assert all(math.isfinite(x) for row in U + V for x in row)
    print("이 손실은 고정된 학습 쌍에 대한 값이며, 검증 성능/실제 추천 성능이 아님.")

    print("\n[6] 중심용 학습 벡터로 항목 간 추천")
    for query in ("t01", "t05", "t09"):
        q = vocabulary.index(query)
        ranking = sorted(((cosine(U[q], U[i]), song) for i, song in enumerate(vocabulary)
                          if song != query), reverse=True)[:3]
        print(f"질의 {query}: {SONGS[query][0]} ({SONGS[query][1]})")
        for similarity, song in ranking:
            print(f"  {song}: {SONGS[song][0]} | {SONGS[song][1]} | cosine={similarity:.6f}")
    print("장르 이름과 제목은 표시용이며 학습에 사용하지 않았습니다.")
    print("원리를 보여주는 인위적 데이터입니다. 새 곡/일반 사용자에 대한 평가가 아닙니다.")
    print("\n검산 통과: UTF-8 복원, BPE 복원, 마스크 평균, 코사인, 양성/음성 경사 방향, 유한값")


def positive_int(value):
    n = int(value)
    if n < 1:
        raise argparse.ArgumentTypeError("1 이상의 정수여야 합니다.")
    return n


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42, help="난수 seed (기본 42)")
    parser.add_argument("--epochs", type=positive_int, default=60, help="학습 반복 수 (기본 60)")
    parser.add_argument("--dims", type=positive_int, default=8, help="벡터 차원 (기본 8)")
    parser.add_argument("--window", type=positive_int, default=2, help="좌우 문맥 창 (기본 2)")
    parser.add_argument("--negatives", type=positive_int, default=3, help="양성당 음성 수 (기본 3)")
    run(parser.parse_args())


if __name__ == "__main__":
    main()

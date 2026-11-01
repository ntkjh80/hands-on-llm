#!/usr/bin/env python3
"""10장: 표준 라이브러리만 쓰는 작은 문장 임베딩 대조 학습.

실제 Transformer/SBERT 학습이 아닙니다. 단어 lookup 벡터 -> 평균 ->
L2 정규화의 작은 공유 인코더를 MNR 손실로 실제 최적화합니다.
보고하는 retrieval은 훈련 예를 다시 찾는 값이며 일반화 성능이 아닙니다.
"""
import argparse
import math
import random
from collections import Counter

PAIRS = [
    ("암호 재설정 방법", "비밀번호 변경 절차"),
    ("환불 처리 기간", "취소금 입금 시간"),
    ("배송 조회 방법", "택배 추적 절차"),
    ("계정 삭제 요청", "회원 탈퇴 신청"),
    ("주소 변경 방법", "배송지 수정 절차"),
    ("영수증 발급 요청", "구매증빙 출력 신청"),
]


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def softmax(xs):
    m = max(xs)
    exp = [math.exp(x - m) for x in xs]
    total = sum(exp)
    return [x / total for x in exp]


def logsumexp(xs):
    m = max(xs)
    return m + math.log(sum(math.exp(x - m) for x in xs))


def normalize(vector):
    norm = math.sqrt(dot(vector, vector))
    if norm < 1e-12:
        raise ValueError("영벡터를 정규화할 수 없습니다.")
    return [x / norm for x in vector], norm


def encode(tokens, weights):
    d = len(next(iter(weights.values())))
    pooled = [sum(weights[t][k] for t in tokens) / len(tokens) for k in range(d)]
    unit, norm = normalize(pooled)
    return unit, norm


def objective(weights, pairs, temperature, want_gradient=False):
    """각 행의 정답 열이 대각선인 한 방향 in-batch MNR."""
    anchors = [a.split() for a, _ in pairs]
    positives = [p.split() for _, p in pairs]
    a_encoded = [encode(ts, weights) for ts in anchors]
    p_encoded = [encode(ts, weights) for ts in positives]
    b = len(pairs)
    d = len(a_encoded[0][0])
    ga = [[0.0] * d for _ in range(b)]
    gp = [[0.0] * d for _ in range(b)]
    loss = 0.0
    similarities = []
    for i, (a, _) in enumerate(a_encoded):
        scores = [dot(a, p) for p, _ in p_encoded]
        similarities.append(scores)
        logits = [s / temperature for s in scores]
        loss += (logsumexp(logits) - logits[i]) / b
        if want_gradient:
            probs = softmax(logits)
            for j, (p, _) in enumerate(p_encoded):
                derivative = (probs[j] - (1.0 if i == j else 0.0)) / (b * temperature)
                for k in range(d):
                    ga[i][k] += derivative * p[k]
                    gp[j][k] += derivative * a[k]
    if not want_gradient:
        return loss, similarities
    grad = {t: [0.0] * d for t in weights}
    for token_lists, encoded, upstream in ((anchors, a_encoded, ga), (positives, p_encoded, gp)):
        for tokens, (unit, norm), g in zip(token_lists, encoded, upstream):
            # d(z/||z||)/dz: (g - u (u dot g)) / ||z||
            along = dot(unit, g)
            pooled_grad = [(g[k] - unit[k] * along) / norm for k in range(d)]
            for token in tokens:
                for k in range(d):
                    grad[token][k] += pooled_grad[k] / len(tokens)
    return loss, similarities, grad


def accuracy(scores):
    return sum(max(range(len(row)), key=lambda j: row[j]) == i for i, row in enumerate(scores)) / len(scores)


def pearson(xs, ys):
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    xc, yc = [x - mx for x in xs], [y - my for y in ys]
    den = math.sqrt(dot(xc, xc) * dot(yc, yc))
    if den == 0:
        raise ValueError("상수 목록의 Pearson은 정의되지 않습니다.")
    return dot(xc, yc) / den


def ranks(xs):
    """동점에는 평균 순위를 부여합니다."""
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    result = [0.0] * len(xs)
    start = 0
    while start < len(xs):
        end = start + 1
        while end < len(xs) and xs[order[end]] == xs[order[start]]:
            end += 1
        average = (start + 1 + end) / 2
        for index in order[start:end]:
            result[index] = average
        start = end
    return result


def deletion_noise(sentence, rate, rng):
    tokens = sentence.split()
    if not tokens:
        raise ValueError("빈 문장은 손상 실습에 사용할 수 없습니다.")
    kept = [token for token in tokens if rng.random() >= rate]
    return " ".join(kept or [rng.choice(tokens)])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--steps", type=int, default=200, help="경사 하강 단계 수 (기본 200)")
    parser.add_argument("--lr", type=float, default=0.12, help="학습률 (기본 0.12)")
    parser.add_argument("--temperature", type=float, default=0.2, help="양수 temperature (기본 0.2)")
    parser.add_argument("--seed", type=int, default=17, help="재현용 난수 seed")
    args = parser.parse_args()
    if args.steps < 0 or not math.isfinite(args.lr) or args.lr <= 0 or not math.isfinite(args.temperature) or args.temperature <= 0:
        parser.error("steps는 0 이상, lr와 temperature는 유한한 양수여야 합니다.")
    rng = random.Random(args.seed)
    vocab = sorted(set(" ".join(s for pair in PAIRS for s in pair).split()))
    weights = {word: [rng.uniform(-0.5, 0.5) for _ in range(6)] for word in vocab}
    print("10장 교육용 계산 실습 | Python 표준 라이브러리 | CPU")
    print("실제 Transformer가 아닌 단어 lookup + 평균 pooling + L2 정규화 인코더입니다.")
    print(f"seed={args.seed}, pairs={len(PAIRS)}, vocabulary={len(vocab)}, dimension=6")
    print(f"temperature={args.temperature}, learning_rate={args.lr}, steps={args.steps}")

    print("\n[1] MNR 손계산: positive와 hard negative의 경쟁")
    for negative in [0.1, 0.6, 0.78]:
        scores = [0.8, negative, -0.1]
        logits = [x / args.temperature for x in scores]
        probability = softmax(logits)[0]
        loss = logsumexp(logits) - logits[0]
        print(f"scores={scores}; positive_probability={probability:.6f}; loss={loss:.6f}")
    collision = logsumexp([4.0, 4.0, -20.0]) - 4.0
    print(f"동일 positive가 2번 들어간 단일 정답 예: loss={collision:.6f}, ln(2)={math.log(2):.6f}")
    assert abs(sum(softmax([1000.0, 999.0])) - 1.0) < 1e-12
    assert collision >= math.log(2) - 1e-12

    print("\n[2] 공유 인코더의 MNR 학습 (훈련 집합 재검색)")
    initial_loss, scores, analytic = objective(weights, PAIRS, args.temperature, True)
    # 독립적인 수치 미분으로 역전파 계산을 한 좌표에서 검산합니다.
    token, coordinate, epsilon = vocab[0], 0, 1e-6
    original = weights[token][coordinate]
    weights[token][coordinate] = original + epsilon
    plus = objective(weights, PAIRS, args.temperature)[0]
    weights[token][coordinate] = original - epsilon
    minus = objective(weights, PAIRS, args.temperature)[0]
    weights[token][coordinate] = original
    numeric = (plus - minus) / (2 * epsilon)
    difference = abs(numeric - analytic[token][coordinate])
    print(f"기울기 검산 | analytic={analytic[token][coordinate]:.8f}, finite_difference={numeric:.8f}, error={difference:.2e}")
    assert difference < 1e-6, "기울기 검산 실패"
    print(f"step=   0 loss={initial_loss:.6f} training_top1={accuracy(scores):.3f}")
    milestones = {1, 10, 50, 100, args.steps}
    for step in range(1, args.steps + 1):
        _, _, gradients = objective(weights, PAIRS, args.temperature, True)
        for word in weights:
            for k in range(6):
                weights[word][k] -= args.lr * gradients[word][k]
        if step in milestones:
            loss, scores = objective(weights, PAIRS, args.temperature)
            print(f"step={step:4d} loss={loss:.6f} training_top1={accuracy(scores):.3f}")
    final_loss, scores = objective(weights, PAIRS, args.temperature)
    print("최종 훈련 질의별 최상위 후보:")
    for i, row in enumerate(scores):
        j = max(range(len(row)), key=lambda k: row[k])
        print(f"  {PAIRS[i][0]} -> {PAIRS[j][1]} (cos={row[j]:.4f}, correct={i == j})")
    print("주의: 위 수치는 6개 훈련 쌍을 다시 찾은 결과이며 독립 평가 성능이 아닙니다.")

    print("\n[3] 평가 점수: 상관과 보정은 다릅니다")
    gold = [0.0, 0.2, 0.6, 0.8]
    predicted = [0.1, 0.3, 0.7, 0.9]
    mse = sum((x - y) ** 2 for x, y in zip(gold, predicted)) / len(gold)
    print(f"gold={gold}; prediction={predicted}")
    print(f"Pearson={pearson(gold, predicted):.6f}; Spearman={pearson(ranks(gold), ranks(predicted)):.6f}; MSE={mse:.6f}")
    assert abs(pearson(gold, predicted) - 1.0) < 1e-12 and mse > 0

    print("\n[4] TSDAE 데이터 구성: 삭제 입력과 복원 정답")
    sentence = "고객은 결제 취소 후 환불 일정을 확인했다"
    for rate in [0.2, 0.5, 0.8]:
        damaged = deletion_noise(sentence, rate, random.Random(args.seed))
        print(f"삭제 확률={rate:.1f} | 입력='{damaged}' | 학습용 정답='{sentence}'")
    print("이 단계는 손상 데이터를 만들 뿐, TSDAE 디코더를 학습하거나 복원 결과를 생성하지 않습니다.")
    print("\n[5] 관계 데이터 점검")
    relation_counts = Counter([0, 0, 1, 2, 0, 1])
    print(f"가상 NLI 라벨 수={dict(sorted(relation_counts.items()))}")
    print(f"MNR의 positive 쌍으로 선택할 entailment 수={relation_counts[0]}")
    print("cosine 회귀 라벨 매핑: entailment=1.0, neutral=0.0, contradiction=0.0")
    print("완료: 수치 검산 통과. 실제 LLM·데이터셋 다운로드·API 호출은 없습니다.")


if __name__ == "__main__":
    main()

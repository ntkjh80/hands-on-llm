#!/usr/bin/env python3
"""4장 교육용 CPU 실습: 고정 특징, 로지스틱 회귀, 코사인, 평가, 출력 검증.

표준 라이브러리만 사용합니다. 실제 언어 모델이나 API를 호출하지 않습니다.
손으로 정한 단어 특징은 문장 임베딩을 대체하지 않으며 성능 수치는 가상
소규모 예제에만 해당합니다. 난수 없이 고정 데이터·초기값을 사용합니다.
"""
from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass

POSITIVE = frozenset("good great delightful enjoyable charming excellent fun moving".split())
NEGATIVE = frozenset("bad boring awful dull disappointing terrible tedious weak".split())

# 직접 작성한 교육용 문장. 실제 영화 리뷰 데이터셋이 아닙니다.
TRAIN = [
    ("good enjoyable movie", 1), ("great charming story", 1),
    ("excellent moving film", 1), ("fun delightful scenes", 1),
    ("good great acting", 1), ("charming enjoyable ending", 1),
    ("bad boring movie", 0), ("awful dull story", 0),
    ("terrible disappointing film", 0), ("tedious weak scenes", 0),
    ("bad awful acting", 0), ("boring disappointing ending", 0),
    ("good but boring", 0), ("boring but good", 1),
]
TEST = [
    ("great fun film", 1), ("delightful moving story", 1),
    ("terrible dull film", 0), ("weak tedious story", 0),
    ("not good", 0), ("not bad", 1),
    ("good but boring", 0), ("boring but good", 1),
    ("a masterpiece", 1), ("a waste", 0),
]


def features(text: str) -> tuple[float, float]:
    """긍정 단어·부정 단어의 비율. 단어 순서·부정어를 의도적으로 버립니다."""
    words = re.findall(r"[a-z]+", text.lower())
    pos = sum(word in POSITIVE for word in words)
    neg = sum(word in NEGATIVE for word in words)
    total = max(1, pos + neg)
    return pos / total, neg / total


def dot(a, b) -> float:
    return sum(x * y for x, y in zip(a, b))


def cosine(a, b) -> float:
    denom = math.sqrt(dot(a, a) * dot(b, b))
    # 0은 이 교육용 실습의 명시적 정책이며 영벡터의 각도가 정의된다는 뜻이 아닙니다.
    return dot(a, b) / denom if denom else 0.0


def sigmoid(z: float) -> float:
    if z >= 0:
        return 1 / (1 + math.exp(-z))
    ez = math.exp(z)
    return ez / (1 + ez)


@dataclass
class Logistic:
    weights: list[float]
    bias: float
    losses: list[tuple[int, float]]

    def probability(self, vector) -> float:
        return sigmoid(dot(self.weights, vector) + self.bias)


def train_logistic(rows, epochs: int, learning_rate: float, l2: float) -> Logistic:
    """전체 배치 경사하강법. 특징 추출기의 사전/규칙은 바꾸지 않습니다."""
    vectors = [features(text) for text, _ in rows]
    labels = [label for _, label in rows]
    w, b, losses = [0.0, 0.0], 0.0, []
    n = len(rows)
    for epoch in range(epochs + 1):
        probs = [sigmoid(dot(w, x) + b) for x in vectors]
        loss = -sum(y * math.log(max(p, 1e-15)) +
                    (1 - y) * math.log(max(1 - p, 1e-15))
                    for y, p in zip(labels, probs)) / n
        loss += l2 * dot(w, w) / 2
        if epoch in {0, epochs // 4, epochs // 2, epochs}:
            losses.append((epoch, loss))
        if epoch == epochs:
            break
        grad = [sum((p - y) * x[j] for x, y, p in zip(vectors, labels, probs)) / n
                + l2 * w[j] for j in range(2)]
        grad_b = sum(p - y for p, y in zip(probs, labels)) / n
        w = [value - learning_rate * g for value, g in zip(w, grad)]
        b -= learning_rate * grad_b
    return Logistic(w, b, losses)


def divide(numerator, denominator) -> float:
    return numerator / denominator if denominator else 0.0


def evaluate(truth, predicted):
    if not truth or len(truth) != len(predicted):
        raise ValueError("정답과 예측은 길이가 같은 비어 있지 않은 목록이어야 합니다.")
    if any(value not in (0, 1) for value in [*truth, *predicted]):
        raise ValueError("평가 라벨은 0 또는 1이어야 합니다.")
    # 행=실제, 열=예측. [[TN, FP], [FN, TP]]
    matrix = [[0, 0], [0, 0]]
    for actual, guess in zip(truth, predicted):
        matrix[actual][guess] += 1
    per_class = []
    for label in (0, 1):
        tp = matrix[label][label]
        support = sum(matrix[label])
        precision = divide(tp, sum(row[label] for row in matrix))
        recall = divide(tp, support)
        f1 = divide(2 * precision * recall, precision + recall)
        per_class.append((precision, recall, f1, support))
    n = len(truth)
    return {
        "matrix": matrix,
        "classes": per_class,
        "accuracy": (matrix[0][0] + matrix[1][1]) / n,
        "macro_f1": sum(row[2] for row in per_class) / 2,
        "weighted_f1": sum(row[2] * row[3] for row in per_class) / n,
    }


def print_report(title, truth, predicted):
    result = evaluate(truth, predicted)
    print(f"\n{title}")
    print("행=실제 0/1, 열=예측 0/1:", result["matrix"])
    print("class  precision  recall     F1  support")
    for label, (p, r, f1, support) in enumerate(result["classes"]):
        print(f"{label:5d}  {p:9.3f}  {r:6.3f}  {f1:5.3f}  {support:7d}")
    print("accuracy={accuracy:.3f}, macro F1={macro_f1:.3f}, weighted F1={weighted_f1:.3f}".format(**result))
    return result


def parse_generated_label(raw: str):
    """허용된 문자열만 매핑. 나머지는 None으로 보류합니다."""
    return {"negative": 0, "positive": 1, "0": 0, "1": 1}.get(raw.strip().lower())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threshold", type=float, default=0.5, help="긍정 결정 임계값, 0~1 (기본 0.5)")
    parser.add_argument("--epochs", type=int, default=800, help="작은 분류기 학습 횟수 (기본 800)")
    args = parser.parse_args()
    if not 0 <= args.threshold <= 1 or args.epochs < 1:
        parser.error("threshold는 0~1, epochs는 1 이상이어야 합니다.")

    print("4장 교육용 계산 실습 | 실제 LLM/API/Rotten Tomatoes 실행 아님")
    print("특징: [긍정 단어 비율, 부정 단어 비율], 고정 사전, 난수 없음")
    print("학습 문장 14개 / 관찰 문장 10개. 장난감 학습 동작 확인용이며 일반화 추정용 분할이 아닙니다.")
    model = train_logistic(TRAIN, epochs=args.epochs, learning_rate=0.4, l2=0.02)
    print("\n1. 고정 특징 위에서 작은 분류기 학습")
    for epoch, loss in model.losses:
        print(f"epoch={epoch:4d}  정규화 포함 train loss={loss:.6f}")
    print(f"w={[round(x, 6) for x in model.weights]}, b={model.bias:.6f}")

    labels = [features("bad boring"), features("good enjoyable")]
    truth, supervised, zero_shot = [], [], []
    print("\n2. 각 문장의 특징·지도학습·라벨 유사도 관찰")
    print("text                         x           p(1)   sup  cos0   cos1  zero true")
    for text, y in TEST:
        vector = features(text)
        p = model.probability(vector)
        sims = [cosine(vector, label) for label in labels]
        # 동점이면 0. 의미가 없을 때도 강제 선택하는 한계를 일부러 드러냅니다.
        zpred = 1 if sims[1] > sims[0] else 0
        pred = int(p >= args.threshold)
        truth.append(y)
        supervised.append(pred)
        zero_shot.append(zpred)
        print(f"{text:28s} {str(tuple(round(v,2) for v in vector)):12s} {p:5.3f}  {pred:3d}  {sims[0]:.3f}  {sims[1]:.3f}  {zpred:4d} {y:4d}")
    print_report(f"지도학습 (threshold={args.threshold:g})", truth, supervised)
    print_report("라벨 설명 특징과 코사인 (라벨 예제 학습 없음)", truth, zero_shot)
    print("주의: label 문자열도 손으로 정한 특징으로 계산합니다. 신경망 문장 임베딩이 아닙니다.")
    print("같은 특징 충돌:", features("good but boring"), features("boring but good"))
    print("부정어 무시:", features("good"), features("not good"))
    print("미등록 어휘:", features("a masterpiece"), "→ 영벡터, 동점 정책에 의해 결정")

    print("\n3. 수학적 검산과 평균의 차이")
    print("cos((0.8,0.6),(1,0)) =", round(cosine((0.8, 0.6), (1, 0)), 3))
    print("cos((8,6),(1,0))     =", round(cosine((8, 6), (1, 0)), 3))
    # TP3 FP1 FN2 TN4를 구성하여 손계산과 독립적으로 대조합니다.
    hand = print_report("손계산 예제 TP=3 FP=1 FN=2 TN=4", [1]*5+[0]*5, [1]*3+[0]*2+[1]+[0]*4)
    assert math.isclose(hand["classes"][1][2], 2 / 3)
    assert math.isclose(cosine((0.8, 0.6), (1, 0)), cosine((8, 6), (1, 0)))
    print_report("불균형: 부정90 긍정10을 전부 부정으로 예측", [0]*90+[1]*10, [0]*100)

    print("\n4. 생성 출력 파싱 (직접 만든 문자열이며 실제 모델 출력 아님)")
    raw_samples = [("positive", 1), ("negative", 0), (" Positive ", 1),
                   ("1", 1), ("0", 0), ("mixed", 0), ("negative.", 0), ("2", 1)]
    valid_truth, valid_pred = [], []
    naive_pred, all_truth = [], []
    for raw, y in raw_samples:
        parsed = parse_generated_label(raw)
        naive = 0 if raw == "negative" else 1
        print(f"raw={raw!r:14s} 허용집합 결과={str(parsed):4s}  else=1 결과={naive} 정답={y}")
        all_truth.append(y)
        naive_pred.append(naive)
        if parsed is not None:
            valid_truth.append(y)
            valid_pred.append(parsed)
    print_report("모든 예상 외 문자를 긍정 처리하는 잘못된 규칙", all_truth, naive_pred)
    print_report("허용 형식만의 분류 성능 (분모 주의)", valid_truth, valid_pred)
    correct = sum(a == b for a, b in zip(valid_truth, valid_pred))
    print(f"coverage={len(valid_pred)}/{len(raw_samples)}={len(valid_pred)/len(raw_samples):.3f}")
    print(f"전체 요청 중 올바른 자동 처리={correct}/{len(raw_samples)}={correct/len(raw_samples):.3f}")
    print("invalid는 긍정/부정으로 몰래 바꾸지 않습니다. 이 예제에서는 재시도 없이 보류합니다.")
    print("\n완료: 손계산 검산 통과. 출력 수치는 이 가상 예제에만 해당합니다.")


if __name__ == "__main__":
    main()

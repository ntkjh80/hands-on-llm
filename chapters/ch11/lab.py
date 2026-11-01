#!/usr/bin/env python3
"""11장 CPU 개념 실습. 표준 라이브러리만 사용하며 실제 BERT/SetFit을 실행하지 않는다.

직접 계산: 작은 표현 모델의 역전파, 동결, 관계 쌍, MLM 대상 선택, BIO 정렬/평가.
교육용 좌표·문장·분할을 사용한다. 이 수치를 실제 언어 모델 성능으로 해석하지 않는다.
"""
import argparse
import itertools
import math
import random


def sigmoid(z):
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    ez = math.exp(z)
    return ez / (1.0 + ez)


def forward(params, x):
    a0, a1, w, b = params
    h = math.tanh(a0 * x[0] + a1 * x[1])
    z = w * h + b
    return h, z, sigmoid(z)


def loss_and_grad(params, data):
    grads = [0.0] * 4
    loss = 0.0
    for x, y in data:
        h, z, p = forward(params, x)
        loss += max(z, 0.0) - y * z + math.log1p(math.exp(-abs(z)))
        dz = p - y
        dh = dz * params[2] * (1.0 - h * h)
        grads[0] += dh * x[0]
        grads[1] += dh * x[1]
        grads[2] += dz * h
        grads[3] += dz
    return loss / len(data), [g / len(data) for g in grads]


def classification_scores(params, data):
    predicted = [int(forward(params, x)[2] >= 0.5) for x, _ in data]
    gold = [y for _, y in data]
    tp = sum(p == y == 1 for p, y in zip(predicted, gold))
    fp = sum(p == 1 and y == 0 for p, y in zip(predicted, gold))
    fn = sum(p == 0 and y == 1 for p, y in zip(predicted, gold))
    acc = sum(p == y for p, y in zip(predicted, gold)) / len(gold)
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    return acc, f1


def gradient_check(params, data):
    _, analytic = loss_and_grad(params, data)
    eps = 1e-6
    errors = []
    for i in range(len(params)):
        plus, minus = params[:], params[:]
        plus[i] += eps
        minus[i] -= eps
        numeric = (loss_and_grad(plus, data)[0] - loss_and_grad(minus, data)[0]) / (2 * eps)
        errors.append(abs(numeric - analytic[i]))
    assert max(errors) < 1e-7, "역전파와 유한차분 검산이 일치하지 않습니다."
    return max(errors)


def freezing_demo(steps, lr):
    print("\n[1] 작은 표현 모델: h=tanh(a·x), p=sigmoid(w*h+b)")
    print("가상 좌표 분류입니다. 정답은 x2의 부호이며 초기 표현은 x1만 봅니다.")
    train = [((x1, x2), int(x2 > 0)) for x1 in (-1., 1.) for x2 in (-1., -.7, .7, 1.)]
    test = [((x1, x2), int(x2 > 0)) for x1 in (-1.5, -.4, .4, 1.5) for x2 in (-1.2, -.8, .8, 1.2)]
    initial = [1.0, 0.0, 0.5, 0.0]
    print(f"학습 {len(train)}개 / 별도 좌표 평가 {len(test)}개")
    print(f"기울기 유한차분 최대 오차: {gradient_check(initial, train):.2e}")
    for frozen in (True, False):
        params = initial[:]
        first_loss, _ = loss_and_grad(params, train)
        for _ in range(steps):
            _, grads = loss_and_grad(params, train)
            if frozen:
                grads[0] = grads[1] = 0.0
            params = [p - lr * g for p, g in zip(params, grads)]
        final_loss, _ = loss_and_grad(params, train)
        acc, f1 = classification_scores(params, test)
        label = "표현 동결, 헤드 학습" if frozen else "표현과 헤드 공동 학습"
        print(f"{label}: loss {first_loss:.6f} -> {final_loss:.6f}")
        print(f"  a=({params[0]:.6f}, {params[1]:.6f}), w={params[2]:.6f}, b={params[3]:.6f}")
        print(f"  가상 평가 정확도={acc:.3f}, 양성 F1={f1:.3f}")
        if frozen:
            assert params[:2] == initial[:2], "동결 표현의 파라미터가 변했습니다."
    print("해석: 실제 언어 모델의 성능 예측이 아니라, 표현 업데이트가 경계를 바꾸는 실험입니다.")


def pair_demo(seed, per_class):
    print("\n[2] SetFit의 관계 데이터: 고유 쌍과 샘플링 항목")
    labels = [0] * per_class + [1] * per_class
    pairs = list(itertools.combinations(range(len(labels)), 2))
    positives = [p for p in pairs if labels[p[0]] == labels[p[1]]]
    negatives = [p for p in pairs if labels[p[0]] != labels[p[1]]]
    expected = len(labels) * (len(labels) - 1) // 2
    assert len(positives) + len(negatives) == expected
    rng = random.Random(seed)
    sampled = []
    for i, label in enumerate(labels):
        same = [j for j, other in enumerate(labels) if other == label and i != j]
        different = [j for j, other in enumerate(labels) if other != label]
        for _ in range(20):
            sampled.append(tuple(sorted((i, rng.choice(same)))))
            sampled.append(tuple(sorted((i, rng.choice(different)))))
    print(f"클래스당 {per_class}개 / 총 {len(labels)}개")
    print(f"고유 무순서 쌍: 양성={len(positives)}, 음성={len(negatives)}, 합={len(pairs)}")
    print(f"각 문장당 양성20+음성20 샘플링: 항목={len(sampled)}, 실제 고유 쌍={len(set(sampled))}")
    print("양성/음성은 감성 종류가 아니라 같은/다른 클래스 관계입니다.")


def masked_example(seed, probability, whole_word):
    tokens = ['[CLS]', 'The', 'play', '##ful', 'actor', 'tells', 'a', 'sur', '##prising',
              'story', 'with', 'strong', 'images', 'and', 'an', 'un', '##usual', 'ending', '[SEP]']
    word_ids = [None, 0, 1, 1, 2, 3, 4, 5, 5, 6, 7, 8, 9, 10, 11, 12, 12, 13, None]
    rng = random.Random(seed)
    if whole_word:
        selected_words = {w for w in sorted(set(word_ids) - {None}) if rng.random() < probability}
        chosen = [i for i, w in enumerate(word_ids) if w in selected_words]
    else:
        chosen = [i for i, w in enumerate(word_ids) if w is not None and rng.random() < probability]
    masked = ['[MASK]' if i in chosen else token for i, token in enumerate(tokens)]
    targets = [token if i in chosen else '-100' for i, token in enumerate(tokens)]
    assert all(word_ids[i] is not None for i in chosen)
    if whole_word:
        for w in set(word_ids) - {None}:
            indices = [i for i, word in enumerate(word_ids) if word == w]
            assert all(i in chosen for i in indices) or all(i not in chosen for i in indices)
    return masked, targets, chosen


def masking_demo(seed, probability):
    print("\n[3] MLM 예측 대상 선택: 가상 토큰 분할, 실제 BERT 실행 아님")
    print("단순화를 위해 선택 위치는 전부 [MASK]로 치환합니다. BERT의 전체 교체 정책을 재현하지 않습니다.")
    for whole_word in (False, True):
        masked, targets, chosen = masked_example(seed, probability, whole_word)
        print("단어 전체 마스킹" if whole_word else "토큰 마스킹")
        print("  입력:", ' '.join(masked))
        print("  목표:", ' '.join(targets))
        print(f"  선택된 토큰 {len(chosen)}개 / 일반 토큰 17개 (선택 확률 {probability:.2f})")
    print("확률은 기대 비율입니다. 한 문장에서 항상 정확히 15%가 선택되는 것은 아닙니다.")


def align_labels(word_ids, word_tags, first_only=False):
    aligned = []
    previous = None
    for word_id in word_ids:
        if word_id is None:
            aligned.append('-100')
        elif word_id != previous:
            aligned.append(word_tags[word_id])
        elif first_only:
            aligned.append('-100')
        else:
            tag = word_tags[word_id]
            aligned.append('I-' + tag[2:] if tag.startswith('B-') else tag)
        previous = word_id
    return aligned


def spans(tags):
    """유효한 BIO 배열만 허용하는 엄격한 교육용 추출기. 끝 위치는 exclusive."""
    result = set()
    start, kind = None, None
    for i, tag in enumerate(list(tags) + ['O']):
        if tag == 'O' or tag.startswith('B-'):
            if start is not None:
                result.add((start, i, kind))
            start, kind = (i, tag[2:]) if tag.startswith('B-') else (None, None)
        elif tag.startswith('I-'):
            if start is None or kind != tag[2:]:
                raise ValueError(f"잘못된 BIO 연결: 위치 {i}, 태그 {tag}")
        else:
            raise ValueError(f"알 수 없는 태그: {tag}")
    return result


def entity_metrics(gold_sequences, predicted_sequences):
    if len(gold_sequences) != len(predicted_sequences):
        raise ValueError('문장 수 불일치')
    gold, predicted = set(), set()
    matched_tokens = total = 0
    for doc_id, (gs, ps) in enumerate(zip(gold_sequences, predicted_sequences)):
        if len(gs) != len(ps):
            raise ValueError('시퀀스 길이 불일치')
        gold.update((doc_id,) + span for span in spans(gs))
        predicted.update((doc_id,) + span for span in spans(ps))
        matched_tokens += sum(g == p for g, p in zip(gs, ps))
        total += len(gs)
    tp = len(gold & predicted)
    fp, fn = len(predicted - gold), len(gold - predicted)
    precision = tp / len(predicted) if predicted else 0.0
    recall = tp / len(gold) if gold else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    return matched_tokens / total if total else 0.0, precision, recall, f1, tp, fp, fn


def ner_demo():
    print("\n[4] BIO 정렬과 개체 경계 평가")
    tokens = ['[CLS]', 'Min', '##su', 'works', 'at', 'Nova', 'Labs', '[SEP]']
    word_ids = [None, 0, 0, 1, 2, 3, 4, None]
    word_tags = ['B-PER', 'O', 'O', 'B-ORG', 'I-ORG']
    aligned = align_labels(word_ids, word_tags)
    print('토큰:', ' '.join(tokens))
    print('모든 조각 학습:', ' '.join(aligned))
    print('첫 조각만 학습:', ' '.join(align_labels(word_ids, word_tags, first_only=True)))
    assert aligned == ['-100', 'B-PER', 'I-PER', 'O', 'O', 'B-ORG', 'I-ORG', '-100']
    gold = [['B-PER', 'I-PER', 'O', 'O', 'B-ORG', 'I-ORG']]
    predicted = [['B-PER', 'O', 'O', 'O', 'B-ORG', 'I-ORG']]
    acc, precision, recall, f1, tp, fp, fn = entity_metrics(gold, predicted)
    print('정답 개체:', sorted(spans(gold[0])))
    print('예측 개체:', sorted(spans(predicted[0])))
    print(f'토큰 정확도={acc:.3f}; 개체 TP={tp}, FP={fp}, FN={fn}')
    print(f'개체 정밀도={precision:.3f}, 재현율={recall:.3f}, F1={f1:.3f}')
    # 사람이 손으로 확인한 두 개체 중 조직 하나만 정확히 맞힌 상황.
    assert (tp, fp, fn) == (1, 1, 1) and math.isclose(f1, 0.5)
    try:
        spans(['I-PER'])
    except ValueError:
        print('문장을 토큰별 길이 1 목록으로 쪼개면 I-PER의 시작 경계를 잃습니다.')
    print('이 평가는 가상 서브토큰 구간 기준입니다. CoNLL 점수와 비교하려면 원래 단어 단위를 맞추세요.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=21)
    parser.add_argument('--steps', type=int, default=300)
    parser.add_argument('--learning-rate', type=float, default=0.15)
    parser.add_argument('--per-class', type=int, default=16)
    parser.add_argument('--mask-probability', type=float, default=0.15)
    args = parser.parse_args()
    if args.steps < 1 or args.learning_rate <= 0 or args.per_class < 2 or not 0 <= args.mask_probability <= 1:
        parser.error('steps>=1, learning-rate>0, per-class>=2, 0<=mask-probability<=1 조건이 필요합니다.')
    print('11장 표현 모델 미세 조정: 표준 Python CPU 개념 실습')
    print(f'seed={args.seed}, steps={args.steps}, learning_rate={args.learning_rate}')
    freezing_demo(args.steps, args.learning_rate)
    pair_demo(args.seed, args.per_class)
    masking_demo(args.seed, args.mask_probability)
    ner_demo()
    print('\n모든 계산과 의미 검산을 완료했습니다. BERT/SetFit/GPU 학습은 실행하지 않았습니다.')


if __name__ == '__main__':
    main()

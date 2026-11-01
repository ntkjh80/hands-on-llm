#!/usr/bin/env python3
"""Chapter 5: transparent CPU experiments, standard library only.

This is an educational DBSCAN + c-TF-IDF + MMR calculation, NOT a
reimplementation or execution of UMAP, HDBSCAN, BERTopic, or an LLM.
Coordinates and keyword embeddings are deliberately hand-designed.
"""
from __future__ import annotations
import argparse
import math
from collections import Counter, deque
from dataclasses import dataclass


@dataclass(frozen=True)
class Document:
    id: str
    text: str
    point: tuple[float, float]


DOCUMENTS = [
    Document('S1', 'speech audio model the', (0.0, 0.0)),
    Document('S2', 'speech audio the', (0.2, 0.1)),
    Document('S3', 'speech audio model', (-0.1, 0.2)),
    Document('S4', 'speech the', (0.1, -0.2)),
    Document('T1', 'translation language model the', (2.0, 0.0)),
    Document('T2', 'translation language the', (2.2, 0.1)),
    Document('T3', 'translation language model', (1.9, 0.2)),
    Document('T4', 'translation the', (2.1, -0.2)),
    Document('H1', 'clinical patient model the', (1.0, 2.0)),
    Document('H2', 'clinical patient the', (1.2, 2.1)),
    Document('H3', 'clinical patient model', (0.9, 2.2)),
    Document('H4', 'clinical the', (1.1, 1.8)),
    Document('X1', 'niche quantum discourse', (4.5, 3.8)),
    Document('X2', 'unusual music notation', (-2.5, 3.5)),
]


def cosine(a, b):
    if len(a) != len(b):
        raise ValueError('Vector dimensions differ')
    na, nb = math.sqrt(sum(x*x for x in a)), math.sqrt(sum(x*x for x in b))
    if na == 0 or nb == 0:
        raise ValueError('Cosine is undefined for a zero vector')
    return sum(x*y for x, y in zip(a, b)) / (na * nb)


def dbscan(points, eps=0.45, min_points=3):
    """Deterministic O(N²) DBSCAN; neighborhood includes the point itself.

    A previously marked noise point can later become a border point.
    HDBSCAN's hierarchy/stability selection is intentionally absent.
    """
    n = len(points)
    neighbors = [
        [j for j in range(n) if math.dist(points[i], points[j]) <= eps]
        for i in range(n)
    ]
    labels = [None] * n
    visited = set()
    cluster_id = 0
    for i in range(n):
        if i in visited:
            continue
        visited.add(i)
        if len(neighbors[i]) < min_points:
            labels[i] = -1
            continue
        labels[i] = cluster_id
        queue = deque(neighbors[i])
        queued = set(neighbors[i])
        while queue:
            j = queue.popleft()
            if j not in visited:
                visited.add(j)
                if len(neighbors[j]) >= min_points:
                    for k in neighbors[j]:
                        if k not in queued:
                            queue.append(k)
                            queued.add(k)
            if labels[j] in (None, -1):
                labels[j] = cluster_id
        cluster_id += 1
    return labels


def class_tfidf(texts, labels, drop_stopwords=False):
    """Exclude label -1 from the calculation in this educational convention.

    BERTopic may retain an outlier representation; this convention is explicit
    so that unrelated noise is not taught as a meaningful topic.
    """
    counts = {}
    for text, label in zip(texts, labels):
        if label == -1:
            continue
        counts.setdefault(label, Counter())
        words = text.lower().split()
        if drop_stopwords:
            words = [w for w in words if w not in {'the'}]
        counts[label].update(words)
    if not counts:
        return {}, {}, {}, 0.0
    totals = {c: sum(row.values()) for c, row in counts.items()}
    if any(total == 0 for total in totals.values()):
        raise ValueError('A class is empty after tokenization')
    global_counts = sum(counts.values(), Counter())
    average_length = sum(totals.values()) / len(counts)
    idf = {w: math.log1p(average_length / f) for w, f in global_counts.items()}
    scores = {
        c: {w: f/totals[c] * idf[w] for w, f in row.items()}
        for c, row in counts.items()
    }
    return scores, counts, idf, average_length


CANDIDATES = {
    'summarization': (1.0, 0.0, 0.0),
    'summary': (0.99, 0.02, 0.0),
    'summaries': (0.98, -0.02, 0.0),
    'document': (0.80, 0.60, 0.0),
    'extractive': (0.75, 0.0, 0.66),
    'rouge': (0.55, 0.45, 0.70),
}


def mmr(candidates, topic, diversity, top_k=4):
    """Select first by relevance, then by relevance minus max redundancy."""
    selected = []
    trace = []
    remaining = list(candidates)
    while remaining and len(selected) < top_k:
        scored = []
        for word in remaining:
            relevance = cosine(candidates[word], topic)
            redundancy = max((cosine(candidates[word], candidates[s])
                              for s in selected), default=0.0)
            score = ((1-diversity)*relevance-diversity*redundancy
                     if selected else relevance)
            scored.append((word, score, relevance, redundancy))
        best = max(scored, key=lambda r: r[1])  # insertion order breaks ties
        selected.append(best[0])
        trace.append(best)
        remaining.remove(best[0])
    return selected, trace


def run_checks():
    # Semantic edge cases and a hand-calculated expected value, not mirror tests.
    labels = dbscan([d.point for d in DOCUMENTS])
    assert Counter(labels) == {0: 4, 1: 4, 2: 4, -1: 2}
    # First point is initially noise but becomes a border point of a later core.
    assert dbscan([(0., 0.), (.9, 0.), (1.8, 0.)], 1.0, 3) == [0, 0, 0]
    assert dbscan([(0., 0.), (10., 10.)], 0.1, 3) == [-1, -1]
    scores, counts, idf, avg = class_tfidf([d.text for d in DOCUMENTS], labels)
    assert avg == 12
    assert math.isclose(scores[0]['speech'], math.log(4)/3)
    assert all(sum(row.values()) == 12 for row in counts.values())
    assert math.isclose(cosine((1., 0.), (2., 0.)), 1.)
    assert math.isclose(cosine((1., 0.), (0., 1.)), 0.)
    low, _ = mmr(CANDIDATES, (1, 0, 0), 0, 3)
    high, _ = mmr(CANDIDATES, (1, 0, 0), 0.7, 3)
    assert low == ['summarization', 'summary', 'summaries']
    assert len(set(high)) == 3 and high != low
    assert class_tfidf(['noise'], [-1]) == ({}, {}, {}, 0.0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--eps', type=float, default=.45,
                        help='DBSCAN neighborhood radius (default: 0.45)')
    parser.add_argument('--min-points', type=int, default=3,
                        help='Core neighborhood count including self (default: 3)')
    parser.add_argument('--diversity', type=float, default=.7,
                        help='MMR diversity in [0, 1] (default: 0.7)')
    parser.add_argument('--drop-stopwords', action='store_true',
                        help='Remove the word "the" before c-TF-IDF calculation')
    parser.add_argument('--check', action='store_true',
                        help='Run meaningful calculation and boundary checks first')
    args = parser.parse_args()
    if not math.isfinite(args.eps) or args.eps <= 0:
        parser.error('--eps must be finite and positive')
    if args.min_points < 1:
        parser.error('--min-points must be positive')
    if not math.isfinite(args.diversity) or not 0 <= args.diversity <= 1:
        parser.error('--diversity must be in [0, 1]')
    if args.check:
        run_checks()
        print('Calculation and boundary checks: PASS')
    print('Chapter 5 educational CPU lab: hand-designed coordinates and vectors')
    print('No UMAP, HDBSCAN, embedding model, BERTopic, or LLM was executed.\n')
    labels = dbscan([d.point for d in DOCUMENTS], args.eps, args.min_points)
    print(f'DBSCAN eps={args.eps:.2f}, min_points={args.min_points}')
    for doc, label in zip(DOCUMENTS, labels):
        print(f'  {doc.id}: label={label:2d}  {doc.text}')
    n_topics = len(set(labels) - {-1})
    n_noise = labels.count(-1)
    print(f'Topics excluding -1: {n_topics}; unique labels: {len(set(labels))}')
    print(f'Noise: {n_noise}/{len(labels)} = {n_noise/len(labels):.1%}\n')
    scores, counts, idf, avg = class_tfidf(
        [d.text for d in DOCUMENTS], labels, args.drop_stopwords)
    print('c-TF-IDF: noise excluded; lowercase whitespace tokenization')
    print(f'Average class length A = {avg:.2f}; drop_stopwords={args.drop_stopwords}')
    if not scores:
        print('No non-noise clusters: no topic representation can be computed.')
    for label, row in scores.items():
        print(f'  Topic {label}: tokens={sum(counts[label].values())}')
        for word, value in sorted(row.items(), key=lambda kv: (-kv[1], kv[0])):
            print(f'    {word:12s} f={counts[label][word]:2d}'
                  f' idf={idf[word]:.6f} weight={value:.6f}')
    print('\nMMR: separate toy keyword vectors for a summarization topic')
    for diversity in [0.0, args.diversity]:
        selected, trace = mmr(CANDIDATES, (1, 0, 0), diversity)
        print(f'  diversity={diversity:.2f}: {", ".join(selected)}')
        for step, (word, score, relevance, redundancy) in enumerate(trace, 1):
            print(f'    step={step} {word:13s} rel={relevance:.4f}'
                  f' max_sim={redundancy:.4f} score={score:.4f}')
    print('\nInterpretation: topic membership, keyword weights, and keyword diversity')
    print('are separate decisions. A generated topic label is NOT produced by this lab.')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Chapter 8: deterministic, standard-library retrieval and grounded extraction lab.

The semantic features and reranker are handwritten teaching rules, NOT trained
embeddings, a cross-encoder, an LLM, or evidence of real model performance.
All corpus facts are fictional. AP@k uses the total number of relevant documents.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import math
import re
from typing import Iterable


@dataclass(frozen=True)
class Document:
    id: str
    text: str


DOCUMENTS = [
    Document('D1', 'Aurora is a science fiction film about space travel.'),
    Document('D2', 'Astronomers praised Aurora for its scientific accuracy and realistic physics.'),
    Document('D3', 'Aurora earned 42 million credits in worldwide ticket revenue.'),
    Document('D4', 'Aurora premiered on 12 April 2030 in the city of Luma.'),
    Document('D5', 'The cast includes Mira Sol and Ren Vale.'),
    Document('D6', 'Product AX-204 receives security updates until June 2031.'),
    Document('D7', 'Product AX-240 receives security updates until June 2029.'),
    Document('D8', 'A physicist advised the Aurora production on gravity.'),
]
QUERIES = [
    ('Q1', 'how precise was the science', {'D2', 'D8'}),
    ('Q2', 'how much money did Aurora make', {'D3'}),
    ('Q3', 'AX-204 security updates', {'D6'}),
    ('Q4', 'mass of the moon', set()),
]
STOP = set('a an the is was for of its and on in to how much did about until'.split())
# Six explicitly named, handcrafted dimensions. Same topic != correct answer.
CONCEPTS = [
    {'science', 'scientific', 'astronomers', 'physics', 'physicist', 'gravity', 'space'},
    {'precise', 'accuracy', 'realistic', 'praised', 'advised'},
    {'money', 'earned', 'credits', 'revenue', 'income', 'ticket'},
    {'premiered', 'date', 'release', 'april', '2030'},
    {'cast', 'mira', 'sol', 'ren', 'vale'},
    {'product', 'security', 'updates', 'support', 'june', '2031', '2029'},
]


def tokens(text: str) -> list[str]:
    """Keep product IDs as complete tokens; a deliberately small English tokenizer."""
    return [t for t in re.findall(r'[a-z0-9]+(?:-[a-z0-9]+)*', text.lower()) if t not in STOP]


def features(text: str) -> tuple[float, ...]:
    counts = Counter(tokens(text))
    vec = tuple(float(sum(counts[w] for w in words)) for words in CONCEPTS)
    norm = math.sqrt(sum(x * x for x in vec))
    return tuple(x / norm for x in vec) if norm else vec


def cosine(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    """Inputs already normalized; zero vectors produce zero similarity."""
    return sum(x * y for x, y in zip(a, b))


def ranked(scores: dict[str, float]) -> list[tuple[str, float]]:
    return sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))


def bm25(query: str, docs: list[Document], k1: float = 1.5, b: float = 0.75) -> list[tuple[str, float]]:
    """BM25 variant with positive IDF log(1 + (N-df+.5)/(df+.5))."""
    bags = [Counter(tokens(d.text)) for d in docs]
    lengths = [sum(bag.values()) for bag in bags]
    avgdl = sum(lengths) / len(lengths)
    df = Counter(word for bag in bags for word in bag)
    scores = {}
    for doc, bag, length in zip(docs, bags, lengths):
        score = 0.0
        for word in set(tokens(query)):
            frequency = bag[word]
            idf = math.log(1 + (len(docs) - df[word] + 0.5) / (df[word] + 0.5))
            denom = frequency + k1 * (1 - b + b * length / avgdl)
            score += idf * frequency * (k1 + 1) / denom
        scores[doc.id] = score
    return ranked(scores)


def dense(query: str, docs: list[Document]) -> list[tuple[str, float]]:
    q = features(query)
    return ranked({doc.id: cosine(q, features(doc.text)) for doc in docs})


def reciprocal_rank_fusion(rankings: list[list[tuple[str, float]]], candidate_count: int, c: int = 60) -> list[tuple[str, float]]:
    """Fuse only the selected top candidate_count items from each ranking."""
    scores: dict[str, float] = {}
    for order in rankings:
        for position, (doc_id, _) in enumerate(order[:candidate_count], 1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1 / (c + position)
    return ranked(scores)


def rule_rerank(query: str, candidates: Iterable[str], docs: list[Document]) -> list[tuple[str, float]]:
    """A transparent rule: semantic feature overlap + lexical match + exact-ID bonus.

    This consumes no qrels. It is NOT a learned cross-encoder.
    """
    by_id = {doc.id: doc for doc in docs}
    q_tokens = set(tokens(query))
    ids = {word for word in q_tokens if re.fullmatch(r'ax-\d+', word)}
    scores = {}
    for doc_id in candidates:
        doc_tokens = set(tokens(by_id[doc_id].text))
        score = cosine(features(query), features(by_id[doc_id].text))
        score += 0.15 * len(q_tokens & doc_tokens) / max(1, len(q_tokens))
        if ids:
            score += 1.0 if ids <= doc_tokens else -1.0
        scores[doc_id] = score
    return ranked(scores)


def metrics(order: list[str], relevant: set[str], k: int) -> dict[str, float | None]:
    """Truncated AP denominator is total relevant, never only retrieved relevant."""
    selected = order[:k]
    found = 0
    ap_sum = 0.0
    for position, doc_id in enumerate(selected, 1):
        if doc_id in relevant:
            found += 1
            ap_sum += found / position
    return {
        'precision': found / k,
        'recall': found / len(relevant) if relevant else None,
        'ap': ap_sum / len(relevant) if relevant else None,
    }


def oracle_ap_ceiling(candidates: list[str], relevant: set[str], k: int) -> float | None:
    """Gold labels only for an upper-bound diagnostic, NEVER the real reranker."""
    if not relevant:
        return None
    available = len(set(candidates) & relevant)
    return min(k, available) / len(relevant)


def chunk_words(text: str, size: int, overlap: int) -> list[str]:
    if size < 1 or overlap < 0 or overlap >= size:
        raise ValueError('require size >= 1 and 0 <= overlap < size')
    words = text.split()
    output = []
    for start in range(0, len(words), size - overlap):
        output.append(' '.join(words[start:start + size]))
        if start + size >= len(words):
            break
    return output


def extract_candidate_excerpts(query: str, order: list[tuple[str, float]], docs: list[Document], threshold: float) -> str:
    """Return verbatim retrieved evidence, not generated prose or factual validation."""
    by_id = {doc.id: doc for doc in docs}
    if not any(features(query)):
        return '근거 부족: 교육용 특징 사전에 질의의 개념이 없습니다.'
    evidence = [f'[{doc_id}] {by_id[doc_id].text}' for doc_id, _ in order
                if cosine(features(query), features(by_id[doc_id].text)) >= threshold]
    return '\n'.join(evidence) if evidence else '근거 부족: 지정한 유사도 기준을 통과한 문서가 없습니다.'


def validate_math() -> None:
    # Independently known textbook cases, including a missed relevant document.
    assert math.isclose(metrics(['A', 'X', 'B'], {'A', 'B'}, 3)['ap'], 5 / 6)
    assert math.isclose(metrics(['A', 'X', 'Y'], {'A', 'B'}, 3)['ap'], 1 / 2)
    assert math.isclose(metrics(['X', 'Y', 'A'], {'A'}, 3)['ap'], 1 / 3)
    assert metrics(['X'], set(), 1)['ap'] is None
    assert oracle_ap_ceiling(['A', 'X'], {'A', 'B'}, 3) == 0.5
    assert tokens('AX-204') == ['ax-204']
    assert chunk_words('a b c d e', 3, 1) == ['a b c', 'c d e']
    assert math.isclose(sum(x*x for x in features('scientific accuracy')), 1.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--top-k', type=int, default=3, help='final result count, 1..8')
    parser.add_argument('--candidates', type=int, default=5, help='per-retriever shortlist, 1..8')
    parser.add_argument('--threshold', type=float, default=0.55, help='handcrafted cosine cutoff, 0..1')
    parser.add_argument('--chunk-size', type=int, default=8)
    parser.add_argument('--overlap', type=int, default=2)
    args = parser.parse_args()
    if not 1 <= args.top_k <= len(DOCUMENTS) or not 1 <= args.candidates <= len(DOCUMENTS):
        parser.error('top-k and candidates must be in 1..8')
    if not 0 <= args.threshold <= 1:
        parser.error('threshold must be in 0..1')
    if args.chunk_size < 1 or not 0 <= args.overlap < args.chunk_size:
        parser.error('require chunk-size >= 1 and 0 <= overlap < chunk-size')
    validate_math()
    print('8장 검색 실습: 모든 문서 사실은 가상이며 LLM/API/학습 모델을 사용하지 않습니다.')
    print('수작업 6차원 특징, BM25, 순위 결합, 규칙 재순위화의 계산 실험입니다.')
    print('결정적 계산: 난수를 사용하지 않으므로 seed가 필요하지 않습니다.')
    print(f'k={args.top_k}, 후보={args.candidates}, 임계값={args.threshold:.2f}')
    print('AP@k 분모=전체 관련 문서 수. 정답 없는 Q4는 MAP에서 제외합니다.')
    for doc in DOCUMENTS:
        print(f'{doc.id}: {doc.text}')
    aps: dict[str, list[float]] = {key: [] for key in ['BM25', '특징 벡터', 'RRF', '규칙 재순위화']}
    for qid, query, relevant in QUERIES:
        print(f'\n{qid}: {query}; 정답={sorted(relevant)}')
        lexical, semantic = bm25(query, DOCUMENTS), dense(query, DOCUMENTS)
        fused = reciprocal_rank_fusion([lexical, semantic], args.candidates)
        candidates = [doc_id for doc_id, _ in fused[:args.candidates]]
        reranked = rule_rerank(query, candidates, DOCUMENTS)
        for name, order in [('BM25', lexical), ('특징 벡터', semantic), ('RRF', fused), ('규칙 재순위화', reranked)]:
            result = metrics([doc_id for doc_id, _ in order], relevant, args.top_k)
            shown = ', '.join(f'{doc_id}:{score:.4f}' for doc_id, score in order[:args.top_k])
            if relevant:
                aps[name].append(result['ap'])
                stats = f"P@k={result['precision']:.4f} R@k={result['recall']:.4f} AP@k={result['ap']:.4f}"
            else:
                stats = '정답 없음: AP 정의 대상에서 제외; 반환 결과가 있어도 답 근거 아님'
            print(f'  {name:8s} [{shown}] {stats}')
        ceiling = oracle_ap_ceiling(candidates, relevant, args.top_k)
        if ceiling is not None:
            print(f'  후보={candidates}; 이 후보만 쓰는 이상적 재순위화 AP@k 상한={ceiling:.4f}')
        print('  근거 후보 발췌(관련성 검증 및 문장 생성 아님):')
        print(extract_candidate_excerpts(query, reranked[:args.top_k], DOCUMENTS, args.threshold))
    print('\n3개 answerable 질의의 MAP@k: 이 가상 자료에서만 유효')
    for name, values in aps.items():
        print(f'  {name}: {sum(values) / len(values):.4f}')
    sample = 'Product Aurora launches domestically in April 2030 but overseas availability depends on certification approval.'
    chunks = chunk_words(sample, args.chunk_size, args.overlap)
    print(f'\n청킹: size={args.chunk_size}, overlap={args.overlap}; 공백 단어 기준이며 모델 토큰이 아님')
    for index, chunk in enumerate(chunks, 1):
        print(f'  C{index}: {chunk}')
    print('\n검산 통과: AP 5/6, 미검색 정답 AP 1/2, 유일 정답 3위 AP 1/3, 청킹 경계.')


if __name__ == '__main__':
    main()

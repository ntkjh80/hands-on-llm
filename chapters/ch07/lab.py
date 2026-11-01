#!/usr/bin/env python3
"""Chapter 7: deterministic CPU demonstrations, not an LLM or live-price service.

Uses only Python's standard library. Word units are whitespace-separated words,
not model tokens. All products, prices, and conversations are teaching data.
"""
from __future__ import annotations
import argparse
import json
import math
import re
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from string import Formatter


def quantize(values, bits):
    """Uniform affine quantization for a small vector; this is not GGUF."""
    lo, hi = min(values), max(values)
    levels = 2 ** bits - 1
    scale = (hi - lo) / levels
    if not scale:
        return list(values), 0.0
    codes = [min(levels, max(0, round((x - lo) / scale))) for x in values]
    restored = [lo + q * scale for q in codes]
    return restored, sum((a - b) ** 2 for a, b in zip(values, restored)) / len(values)


def format_checked(template, values):
    required = {field for _, field, _, _ in Formatter().parse(template) if field}
    missing = required - values.keys()
    if missing:
        raise ValueError('Missing fields: ' + ', '.join(sorted(missing)))
    return template.format_map(values)


def story_chain(summary):
    """Rule-based text construction, intentionally not generated storytelling."""
    state = {'summary': summary}
    trace = []
    steps = [
        ('title', 'A notebook about {summary}'),
        ('character', 'A student investigates the topic in "{title}".'),
        ('story', '{character} The student writes observations about {summary}.'),
    ]
    for key, template in steps:
        value = format_checked(template, state)
        state = {**state, key: value}
        trace.append({'writes': key, 'available_keys': list(state), 'value': value})
    return state, trace


def extract_facts(text):
    """A tiny, disclosed grammar: only these explicit statements are understood."""
    patterns = {
        'name': r'(?:My name is|Your name is) ([A-Za-z]+)',
        'age': r'(?:I am|Your age is) (\d+)(?: years old)?',
        'city': r'(?:I live in|You live in) ([A-Za-z]+)',
    }
    result = {}
    for key, pattern in patterns.items():
        matches = re.findall(pattern, text)
        if matches:
            result[key] = matches[-1]
    return result


def summary_text(facts):
    pieces = []
    if 'name' in facts:
        pieces.append(f"My name is {facts['name']}.")
    if 'age' in facts:
        pieces.append(f"I am {facts['age']} years old.")
    if 'city' in facts:
        pieces.append(f"I live in {facts['city']}.")
    return ' '.join(pieces)


def answer_rule(question, context):
    facts = extract_facts(context + '\n' + question)
    q = question.lower()
    if 'what is my name' in q:
        return f"Your name is {facts['name']}." if 'name' in facts else 'Name unknown.'
    if 'what is my age' in q:
        return f"Your age is {facts['age']}." if 'age' in facts else 'Age unknown.'
    if 'where do i live' in q:
        return f"You live in {facts['city']}." if 'city' in facts else 'City unknown.'
    if '1 + 1' in q:
        return '1 + 1 = 2.'
    if '3 + 3' in q:
        return '3 + 3 = 6.'
    return 'Update received.'


QUESTIONS = [
    'My name is Mina. I am 29 years old. What is 1 + 1?',
    'What is 3 + 3?',
    'What is my name?',
    'What is my age?',
    'Correction: I am 30 years old. I live in Busan.',
    'Where do I live?',
]


def memory_demo(mode, window, budget):
    history, facts, rows = [], {}, []
    for turn, question in enumerate(QUESTIONS, 1):
        selected = history if mode == 'full' else history[-window:] if mode == 'window' else []
        context = summary_text(facts) if mode == 'summary' else '\n'.join(
            f'Human: {q}\nAssistant: {a}' for q, a in selected)
        units = len(context.split())
        answer = answer_rule(question, context)
        rows.append({'turn': turn, 'context_word_units': units,
                     'over_history_budget': units > budget, 'answer': answer,
                     'visible_facts': extract_facts(context), 'context': context})
        history.append((question, answer))
        # Only explicit user facts enter this teaching summary. No LLM is called.
        facts.update(extract_facts(question))
    return rows


CATALOG = {'StudyLaptop': {'usd': '1200.00', 'source': 'local teaching fixture A'},
           'StudyLaptop alternative': {'usd': '1200.00', 'source': 'local teaching fixture B'}}


def agent_demo(rate, fail_first=False, max_steps=5):
    """A deterministic policy demonstrates an execution loop, not an LLM agent."""
    state, trace = {}, []
    attempts = 0
    for step in range(1, max_steps + 1):
        if 'price' not in state:
            action = 'lookup_price'
            query = 'StudyLaptop' if attempts == 0 else 'StudyLaptop alternative'
            attempts += 1
            observation = ({'error': 'simulated lookup failure'} if fail_first and attempts == 1
                           else CATALOG[query].copy())
            if 'usd' in observation:
                state['price'] = observation
            trace.append({'step': step, 'action': action, 'input': query, 'observation': observation})
        elif 'eur' not in state:
            usd = Decimal(state['price']['usd'])
            eur = (usd * rate).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
            state['eur'] = str(eur)
            trace.append({'step': step, 'action': 'convert',
                          'input': {'usd': str(usd), 'eur_per_usd': str(rate)},
                          'observation': {'eur': str(eur)}})
        else:
            result = {'status': 'complete', 'usd': state['price']['usd'], 'eur': state['eur'],
                      'eur_per_usd': str(rate), 'source': state['price']['source']}
            trace.append({'step': step, 'action': 'finish', 'observation': result})
            return trace, result
    return trace, {'status': 'incomplete', 'reason': 'step limit reached', 'state': state}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--window', type=int, default=2, help='recent completed interaction pairs')
    parser.add_argument('--budget', type=int, default=35, help='history word-unit budget, NOT tokens')
    parser.add_argument('--rate', default='0.85', help='fictional EUR per USD exchange rate')
    parser.add_argument('--max-steps', type=int, default=5, help='agent loop limit including finish')
    parser.add_argument('--lookup-fail-first', action='store_true', help='simulate one failed lookup')
    args = parser.parse_args()
    try:
        rate = Decimal(args.rate)
    except InvalidOperation:
        parser.error('--rate must be a finite positive decimal')
    if not rate.is_finite() or rate <= 0:
        parser.error('--rate must be a finite positive decimal')
    if args.window < 1 or args.budget < 1 or args.max_steps < 1:
        parser.error('--window, --budget and --max-steps must be positive')

    print('CH07 | CPU-only teaching simulation | No LLM, API, live search, or real price data')
    print('\n[1] Quantization: uniform affine example, not a GGUF implementation')
    values = [-1.0, -0.73, -0.31, -0.05, 0.13, 0.38, 0.67, 1.0]
    for bits in (2, 4, 8):
        restored, mse = quantize(values, bits)
        print(f'{bits:2d} bit | ideal packed weight bytes={len(values)*bits/8:.1f} | MSE={mse:.8f}')
        print('restored:', ', '.join(f'{x:.4f}' for x in restored))
        assert all(math.isfinite(x) for x in restored)
    print('Storage estimate excludes scales, metadata, activations, KV cache, and runtime overhead.')

    print('\n[2] Chain: rule-based template composition')
    state, trace = story_chain('a missing notebook')
    for row in trace:
        print(json.dumps(row, ensure_ascii=False))
    assert set(state) == {'summary', 'title', 'character', 'story'}
    try:
        format_checked('{summary}: {title}', {'summary': 'topic'})
    except ValueError as error:
        print('Expected boundary error:', error)

    print('\n[3] Memory: exact rule-based responses; counts are WORD UNITS, not tokenizer tokens')
    results = {}
    for mode in ('full', 'window', 'summary'):
        results[mode] = memory_demo(mode, args.window, args.budget)
        print(f'\nMODE={mode}; recent window={args.window}; history budget={args.budget}')
        for row in results[mode]:
            print(f"turn {row['turn']}: history={row['context_word_units']:3d} words | "
                  f"over_budget={str(row['over_history_budget']):5s} | {row['answer']}")
        print('cumulative history word units:', sum(x['context_word_units'] for x in results[mode]))
    assert results['full'][3]['answer'] == 'Your age is 29.'
    assert results['summary'][3]['answer'] == 'Your age is 29.'
    if args.window == 2:
        assert results['window'][2]['answer'] == 'Your name is Mina.'
        assert results['window'][3]['answer'] == 'Age unknown.'
    assert results['summary'][-1]['answer'] == 'You live in Busan.'
    print('Summary is an explicit-fact dictionary; it cannot preserve every original sentence.')
    print('Budget flags report overflow only; this demonstration does not silently truncate text.')

    print('\n[4] Tool execution loop: deterministic policy, fictional data')
    trace, result = agent_demo(rate, args.lookup_fail_first, args.max_steps)
    for row in trace:
        print(json.dumps(row, ensure_ascii=False))
    print('FINAL:', json.dumps(result, ensure_ascii=False))
    if result['status'] == 'complete':
        expected = (Decimal('1200.00') * rate).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
        assert Decimal(result['eur']) == expected
    print('\nMeaningful checks passed: chain fields, memory retention, and completed conversion arithmetic.')


if __name__ == '__main__':
    main()

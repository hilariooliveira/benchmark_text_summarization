from src.basic_classes.classes import Token


def extract_ngrams(tokens: list, n: int) -> list:
    ngrams = []
    tam = len(tokens) - n + 1
    for i in range(tam):
        ngram = concatenate(tokens, i, i + n)
        if ngram:
            ngrams.append(ngram)
    return ngrams


def concatenate(tokens: list, begin: int, end: int) -> Token:
    text = ''
    stem = ''
    ngram = Token()
    for i in range(begin, end):
        text += tokens[i].text + ' '
        stem += tokens[i].stem + ' '
        ngram.add_sub_token(tokens[i])
    ngram.text = text
    ngram.stem = stem
    return ngram


def build_ngrams(tokens: list, min_ngram: int, max_ngram: int) -> list[Token] | None:
    if not tokens:
        return None
    if min_ngram <= 0:
        min_ngram = 1
    if max_ngram > len(tokens):
        max_ngram = len(tokens)
    all_ngrams = []
    for i in range(min_ngram, max_ngram + 1):
        ngrams = extract_ngrams(tokens, i)
        if ngrams is not None:
            all_ngrams.extend(ngrams)
    for id_, ngram in enumerate(all_ngrams, start=1):
        ngram.id = id_
    return all_ngrams

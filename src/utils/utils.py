import re


def get_token_type(text: str, punctuation_symbols: list=[]) -> str:
    if text in punctuation_symbols:
        return 'PUNCT'
    pattern_number = re.compile(r"\\d+(('.'|,)\\d+)?")
    match_number = pattern_number.match(text)
    if match_number is not None and text == match_number.group():
        return 'NUM'
    pattern_symbol = re.compile(r'\W+')
    match_symbol = pattern_symbol.match(text)
    if match_symbol is not None and text == match_symbol.group():
        return 'SYM'
    return 'WORD'


def filter_concepts(concepts: list) -> list:
    new_concepts = []
    for concept in concepts:
        count_stop_words = 0
        for token in concept.sub_tokens:
            if token.is_stop_word:
                count_stop_words += 1
        if count_stop_words != len(concept.sub_tokens):
            new_concepts.append(concept)
    return new_concepts


def save_file(file_path: str, content: str):
    content = content.replace('\x96', '')
    content = content.replace('\x94', '')
    content = content.replace('\ufeff', '')
    content = content.replace('\u0144', '')
    content = content.replace('\u2153', '')
    with open(file_path, 'w') as file:
        file.write(content)

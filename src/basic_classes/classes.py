class Token:
    def __init__(self):
        self.id: int = -1
        self.text: str = ''
        self.stem: str = ''
        self.pos: str = ''
        self.is_stop_word: bool = False
        self.type: str = ''
        self.sub_tokens = []
        self.weight: float = 0
        self.weights: dict[str, float] = {}

    def add_sub_token(self, token):
        self.sub_tokens.append(token)

    def add_weight(self, name: str, value: float):
        self.weights[name] = value

    def __repr__(self) -> str:
        return self.stem

    def __hash__(self) -> float:
        return hash(self.stem)

    def __eq__(self, other) -> bool:
        return isinstance(other, Token) and other.stem == self.stem


class Summary:
    def __init__(self, name: str = None, text: str = None):
        self.name: str = name
        self.text: str = text
        self.concepts: list[Token] = []
        self.features: dict[str, float] = {}
        self.rouge_scores: dict = {}

    def add_feature(self, name: str, value: float):
        self.features[name] = value


class Sentence:
    def __init__(self):
        self.id: int = -1
        self.text: str = ''
        self.tokens: list[Token] = []
        self.tokens_no_stopwords: list[Token] = []
        self.concepts: list[Token] = []

    def add_token(self, token: Token):
        self.tokens.append(token)

    def add_token_no_stopwords(self, token: Token):
        self.tokens_no_stopwords.append(token)

    def __repr__(self) -> str:
        return self.text

    def __hash__(self) -> float:
        return hash(self.text)

    def __eq__(self, other) -> bool:
        return isinstance(other, Sentence) and other.text == self.text


class Document:
    def __init__(self):
        self.id: int = -1
        self.name: str = ''
        self.title: str = ''
        self.text: str = ''
        self.sentences: list[Sentence] = []
        self.tokens_title: list[Token] = []
        self.concepts: list[Token] = []
        self.highlights: str = ''
        self.gold_standard: str = ''
        self.candidate_summaries: list[Summary] = []
        self.references_summaries: list[str] = []
        self.summary: str = ''
        self.category: str = ''

    def add_sentence(self, sentence: Sentence):
        self.sentences.append(sentence)

    def add_reference_summary(self, reference_summary: str):
        self.references_summaries.append(reference_summary)

    def add_candidate_summary(self, candidate_summary: Summary):
        self.candidate_summaries.append(candidate_summary)

    def __repr__(self):
        return self.name


class Corpus:
    def __init__(self, documents: list[Document] = None):
        self.documents: list[Document] = documents

    def add_document(self, document: Document):
        if self.documents is None:
            self.documents = []
        self.documents.append(document)

    def add_documents(self, documents: list[Document]):
        if self.documents is None:
            self.documents = []
        self.documents.extend(documents)

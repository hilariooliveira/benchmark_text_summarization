import os
import xmltodict

from src.basic_classes.classes import Corpus, Document, Summary


def build_cnn_corpus(corpus_dir: str) -> Corpus:
    assert os.path.exists(corpus_dir)
    corpus = Corpus()
    documents_names = os.listdir(corpus_dir)
    for doc_name in documents_names:
        document_file = os.path.join(corpus_dir, doc_name)
        with open(document_file, encoding='utf-8') as file:
            xml_doc = xmltodict.parse(file.read())
        xml_doc = xml_doc['document']
        title = xml_doc['title']
        title = replace_marks(title)
        category = xml_doc['@category']
        highlights_element = xml_doc['summaries']['highlights']
        highlights = ''
        for sentence_highlight in highlights_element['sentence']:
            highlights += sentence_highlight['#text'] + '\n'
        highlights = highlights.strip()
        highlights = replace_marks(highlights)
        gold_standard_element = xml_doc['summaries']['gold_standard']
        gold_standard = ''
        if isinstance(gold_standard_element['sentence'], dict):
            gold_standard += gold_standard_element['sentence']['#text']
        else:
            for sentence_gold_standard in gold_standard_element['sentence']:
                if isinstance(sentence_gold_standard, dict):
                    gold_standard += sentence_gold_standard['#text'] + '\n'
        gold_standard = gold_standard.strip()
        gold_standard = replace_marks(gold_standard)
        article_element = xml_doc['article']
        text = ''
        for paragraph_element in article_element['paragraph']:
            sentences_element = paragraph_element['sentences']['sentence']
            if isinstance(sentences_element, list):
                for sentence_element in sentences_element:
                    text += sentence_element['content'] + ' '
            else:
                text += sentences_element['content'] + ' '
        text = text.strip()
        text = replace_marks(text)
        document = Document()
        doc_name = doc_name.replace('.xml', '').replace("'", '').lower()
        document.name = doc_name.replace(';', '').replace('&', '').replace('%', '').strip()
        document.title = title
        document.highlights = highlights
        document.gold_standard = gold_standard
        document.text = text
        document.category = category
        corpus.add_document(document)
    return corpus


def replace_marks(text: str) -> str:
    text = text.replace('&amp;', '&')
    text = text.replace('&quot;', '"')
    text = text.replace('&apost;', "'")
    return text


def save_summary(document: Document, summaries_dir: str, summary_name: str):
    if document.id is not None and document.id != -1:
        document_dir = os.path.join(summaries_dir, str(document.id))
    else:
        document_dir = os.path.join(summaries_dir, document.name.lower())
    os.makedirs(document_dir, exist_ok=True)
    summary_path = os.path.join(document_dir, summary_name + '.txt')
    with open(summary_path, 'w', encoding='utf-8') as file:
        file.write(document.summary)


def read_summaries(corpus: Corpus, summaries_dir: str) -> None:
    for doc in corpus.documents:
        if doc.id is not None and doc.id != -1:
            doc_dir = os.path.join(summaries_dir, str(doc.id))
        else:
            doc_dir = os.path.join(summaries_dir, doc.name)
        if not os.path.exists(doc_dir):
            continue
        summaries_names = os.listdir(doc_dir)
        for summary_name in summaries_names:
            summary_path = os.path.join(doc_dir, summary_name)
            with open(summary_path, encoding='latin-1') as file:
                text_summary = file.read()
            doc.add_candidate_summary(
                Summary(name=summary_name.replace('.txt', ''), text=text_summary)
            )

import sys

from tqdm import tqdm
from src.corpora import corpora_utils as corpora


def summarize_roberta(documents, tokenizer, model, min_len, max_len, num_beams, summaries_dir, model_name,
                      device='cpu'):
    with tqdm(total=len(documents), colour='red', file=sys.stdout, desc='  Summarizing') as pbar:
        for document in documents:
            max_len = len(document.highlights.split()) if max_len == -1 else max_len
            input_ids = tokenizer(document.text, padding='max_length', truncation=True, max_length=512,
                                  return_tensors='pt').input_ids
            input_ids = input_ids.to(device)
            output_ids = model.generate(input_ids, num_beams=num_beams, min_length=min_len,
                                        max_length=max_len)[0]
            summary = tokenizer.decode(output_ids, skip_special_tokens=True)
            document.summary = summary
            corpora.save_summary(document, summaries_dir, model_name)
            pbar.update(1)


def summarize_bert(documents, tokenizer, model, min_len, max_len, num_beams, summaries_dir, model_name,
                   device='cpu'):
    with tqdm(total=len(documents), colour='red', file=sys.stdout, desc='  Summarizing') as pbar:
        for document in documents:
            inputs = tokenizer(document.text, padding='max_length', truncation=True, max_length=512,
                               return_tensors='pt')
            input_ids = inputs.input_ids.to(device)
            attention_mask = inputs.attention_mask.to(device)
            output = model.generate(input_ids, num_beams=num_beams, min_length=min_len, max_length=max_len,
                                    attention_mask=attention_mask)
            summary = tokenizer.decode(output[0], skip_special_tokens=True)
            document.summary = summary
            corpora.save_summary(document, summaries_dir, model_name)
            pbar.update(1)


def summarize_bart(documents, tokenizer, model, min_len, max_len, num_beams, summaries_dir, model_name,
                   device='cpu'):
    with tqdm(total=len(documents), colour='red', file=sys.stdout, desc='  Summarizing') as pbar:
        for document in documents:
            max_len = len(document.highlights.split()) if max_len == -1 else max_len
            inputs = tokenizer(document.text, truncation=True, padding='longest', max_length=1024,
                               return_tensors='pt').to(device)
            summary_ids = model.generate(inputs['input_ids'], num_beams=num_beams, min_length=min_len,
                                         max_length=max_len, early_stopping=True)
            tokens_summary = [tokenizer.decode(g, skip_special_tokens=True,
                                               clean_up_tokenization_spaces=False) for g in summary_ids]
            summary = ' '.join(tokens_summary).strip()
            document.summary = summary
            corpora.save_summary(document, summaries_dir, model_name)
            pbar.update(1)


def summarize_pegasus(documents, tokenizer, model, min_len, max_len, num_beams, summaries_dir, model_name,
                      device='cpu'):
    with tqdm(total=len(documents), colour='red', file=sys.stdout, desc='  Summarizing') as pbar:
        for document in documents:
            max_len = len(document.highlights.split()) if max_len == -1 else max_len
            batch = tokenizer(document.text, truncation=True, padding='longest', max_length=1024,
                              return_tensors='pt', add_special_tokens=True).to(device)
            translated = model.generate(**batch, min_length=min_len, max_length=max_len, num_beams=num_beams)
            summary = tokenizer.batch_decode(translated, skip_special_tokens=True,
                                             clean_up_tokenization_spaces=True)
            summary = summary[0].replace('<n>', ' ')
            document.summary = summary
            corpora.save_summary(document, summaries_dir, model_name)
            pbar.update(1)


def summarize_flan_t5(documents, tokenizer, model, min_len, max_len, num_beams, summaries_dir, model_name,
                      device='cpu'):
    with tqdm(total=len(documents), colour='red', file=sys.stdout, desc='  Summarizing') as pbar:
        for document in documents:
            input_ids = tokenizer(document.text, padding='max_length', truncation=True, max_length=512,
                                  return_tensors='pt').input_ids
            input_ids = input_ids.to(device)
            output_ids = model.generate(input_ids, num_beams=num_beams, min_length=min_len,
                                        max_length=max_len)[0]
            summary = tokenizer.decode(output_ids, skip_special_tokens=True)
            document.summary = summary
            corpora.save_summary(document, summaries_dir, model_name)
            pbar.update(1)


def summarize_brio(documents, tokenizer, model, min_len, max_len, num_beams, summaries_dir, model_name,
                   device='cpu'):
    with tqdm(total=len(documents), colour='red', file=sys.stdout, desc='  Summarizing') as pbar:
        for document in documents:
            max_len = len(document.highlights.split()) if max_len == -1 else max_len
            inputs = tokenizer(document.text.lower(), truncation=True, padding='longest', max_length=1024,
                               return_tensors='pt').to(device)
            summary_ids = model.generate(inputs['input_ids'], num_beams=num_beams, min_length=min_len,
                                         max_length=max_len, early_stopping=True)
            tokens_summary = [tokenizer.decode(g, skip_special_tokens=True,
                                               clean_up_tokenization_spaces=False) for g in summary_ids]
            summary = ' '.join(tokens_summary).strip()
            document.summary = summary
            corpora.save_summary(document, summaries_dir, model_name)
            pbar.update(1)

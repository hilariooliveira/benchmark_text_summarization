import statistics as st
import os
import re
import json

from src.basic_classes.classes import Document, Corpus
from rouge import Rouge

"""
    https://github.com/Diego999/py-rouge
"""


def evaluate_summaries(document: Document, is_use_stemming: bool, limit_words: int = 100):
    evaluator = Rouge(metrics=['rouge-n', 'rouge-l'], max_n=2, limit_length=True, length_limit=limit_words,
                      length_limit_type='words', apply_avg=True, apply_best=False, alpha=0.5,
                      weight_factor=1.2, stemming=is_use_stemming)
    for summary in document.candidate_summaries:
        if summary.name == 'davinci_003_ext_03':
            summary.text = re.sub(r'\n*\d+\.', '\n', summary.text).strip()
        references_summaries = list(set(document.references_summaries))
        rouge_scores = evaluator.get_scores(summary.text.replace('\n', ' '), references_summaries)
        summary.rouge_scores = rouge_scores


def generate_document_report(document: Document, directory: str):
    avg_scores_configurations = {}
    for summary in document.candidate_summaries:
        rouge_variations = {}
        for variation, metrics in summary.rouge_scores.items():
            rouge_variations[variation] = {'r': metrics['r'], 'p': metrics['p'], 'f': metrics['f']}
        avg_scores_configurations[summary.name] = rouge_variations
    save_report_document(avg_scores_configurations, directory)


def save_report_document(avg_scores_configurations: dict, directory: str):
    report_results = {}
    for configuration, rouge_variations in avg_scores_configurations.items():
        for rouge_variation, metrics_values in rouge_variations.items():
            if rouge_variation in report_results:
                configurations_results = report_results[rouge_variation]
            else:
                configurations_results = {}
            configurations_results[configuration] = metrics_values
            report_results[rouge_variation] = configurations_results
    for rouge_variation, configurations_results in report_results.items():
        results_report_file = os.path.join(directory, rouge_variation + '.csv')
        report = 'System;Recall;Precision;F-measure\n'
        for configuration, results in configurations_results.items():
            report += configuration + ';'
            report += str(results['r']) + ';'
            report += str(results['p']) + ';'
            report += str(results['f']) + '\n'
        report = report[:-1]
        with open(results_report_file, 'w') as file:
            file.write(report)


def generate_report(corpus: Corpus, summaries_dir: str, prefix_file_name: str, metric_name: str):
    avg_scores_configurations = {}
    for document in corpus.documents:
        for summary in document.candidate_summaries:
            rouge_variations = {}
            if summary.name in avg_scores_configurations:
                rouge_variations = avg_scores_configurations[summary.name]
            for variation, metrics in summary.rouge_scores.items():
                if variation in rouge_variations:
                    metrics_values = rouge_variations[variation]
                else:
                    metrics_values = {'r': [], 'p': [], 'f': []}
                metrics_values['r'].append(metrics['r'])
                metrics_values['p'].append(metrics['p'])
                metrics_values['f'].append(metrics['f'])
                rouge_variations[variation] = metrics_values
            avg_scores_configurations[summary.name] = rouge_variations
    save_csv_report(avg_scores_configurations, summaries_dir, prefix_file_name)
    save_json_report(avg_scores_configurations, summaries_dir, prefix_file_name, metric_name)


def save_csv_report(avg_scores_configurations: dict, summaries_dir: str, prefix_file_name: str):
    report_results = {}
    for configuration, rouge_variations in avg_scores_configurations.items():
        for rouge_variation, metrics_values in rouge_variations.items():
            if rouge_variation in report_results:
                configurations_results = report_results[rouge_variation]
            else:
                configurations_results = {}
            configurations_results[configuration] = metrics_values
            report_results[rouge_variation] = configurations_results
    for rouge_variation, configurations_results in report_results.items():
        results_report_file = os.path.join(summaries_dir, f'{rouge_variation}_{prefix_file_name}.csv')
        report = 'System;Recall;Standard Deviation;Precision;Standard Deviation;F-measure;Standard Deviation\n'
        for configuration, results in configurations_results.items():
            report += configuration + ';'
            avg_recall = st.mean(results['r'])
            avg_precision = st.mean(results['p'])
            avg_f_score = st.mean(results['f'])
            if len(results['r']) > 1:
                report += str(avg_recall) + ';'
                report += str(st.stdev(results['r'])) + ';'
                report += str(avg_precision) + ';'
                report += str(st.stdev(results['p'])) + ';'
                report += str(avg_f_score) + ';'
                report += str(st.stdev(results['f'])) + ';'
            else:
                report += str(avg_recall) + ';'
                report += '0;'
                report += str(avg_precision) + ';'
                report += '0;'
                report += str(avg_f_score) + ';'
                report += '0;'
            report += '\n'
        report = report[:-1]
        report = report.replace('.', ',')
        with open(results_report_file, 'w') as file:
            file.write(report)


def save_json_report(avg_scores_configurations: dict, summaries_dir: str, prefix_file_name: str,
                     metric_name: str):

    if len(prefix_file_name) > 0:
        file_name = f'results_{metric_name}_{prefix_file_name}.json'
    else:
        file_name = f'results_{metric_name}.json'

    results_report_file = os.path.join(summaries_dir, file_name)

    with open(file=results_report_file, mode='w') as json_file:
        json.dump(avg_scores_configurations, json_file, indent=4)

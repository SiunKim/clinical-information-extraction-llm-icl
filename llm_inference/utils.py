import os
import random
import pickle
import warnings

from typing import List, Dict, Tuple, Any
from itertools import product

from collections import defaultdict

import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
import numpy as np

from transformers import AutoTokenizer

# from transformers import AutoTokenizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings('ignore', category=UndefinedMetricWarning)

from settings import (
    PPS,
    set_llm_optimal_params
    )



def set_entity_labels_for_entity_group_sentence(entity_group,
                                                sentence_entity):
    """set_entity_labels_for_entity_group_sentence"""
    if sentence_entity:
        entity_labels = PPS.sentence_entity_labels
    elif entity_group:
        entity_labels_for_grouping =  \
            [entity for entities in PPS.entity_groups.values() for entity in entities]
        entity_labels = \
            list(PPS.entity_groups) + \
                [entity for entity in PPS.entity_labels
                    if entity not in entity_labels_for_grouping]
    else:
        entity_labels = PPS.entity_labels
    return entity_labels

def set_task_description(entity_labels: List[str],
                         entity_group: bool,
                         sentence_entity: bool):
    """Set task_description"""
    task_description = ""
    if sentence_entity:
        for i, entity_label in enumerate(PPS.sentence_entity_labels):
            with open(f"{PPS.dir_prompt_formats}/prompt_ner_entity_description_{entity_label}.txt", 'r', encoding='utf-8') as f:
                entity_description = f.read()
            task_description += f"{i + 1}. {entity_description}\n"
    elif entity_group:
        for i, entity_label in enumerate(entity_labels):
            if entity_label in PPS.entity_groups:
                with open(f"{PPS.dir_prompt_formats}/prompt_ner_entity_group_description_{entity_label}.txt", 'r', encoding='utf-8') as f:
                    entity_description = f.read()
            else:
                with open(f"{PPS.dir_prompt_formats}/prompt_ner_entity_description_{entity_label}.txt", 'r', encoding='utf-8') as f:
                    entity_description = f.read()
            task_description += f"{i + 1}. {entity_description}\n"
    else:
        for i, entity_group_name in enumerate(entity_labels):
            with open(f"{PPS.dir_prompt_formats}/prompt_ner_entity_description_{entity_group_name}.txt", 'r', encoding='utf-8') as f:
                entity_description = f.read()
            task_description += f"{i + 1}. {entity_description}\n"

    return  task_description

def set_confusing_cases():
    """Set confusing cases"""
    try:
        with open(f"{PPS.dir_prompt_formats}/prompt_ner_final_annotation_commonlyconfused.txt",
                'r', encoding='utf-8') as f:
            confusing_cases = f.read()
        return confusing_cases
    except FileNotFoundError:
        return None

def create_tagged_text_with_priority(annotation_data: Dict[str, Any],
                                     entity_labels: List[str],
                                     with_abbr: bool=False,
                                     only_for_formatting: bool=False) \
    -> Tuple[str, Dict[Tuple[str, str], int]]:
    """
    Create a tagged text from the annotation data, handling overlapping entities based on priority.
    """
    entities = sorted(annotation_data['entities'],
                      key=lambda x: (x['start'], entity_labels.index(x['label'])))
    final_entities = []
    overlap_counts = defaultdict(int)
    for entity in entities:
        if entity['label'] in entity_labels:
            overlaps = [e for e in final_entities
                        if (e['start'] < entity['end'] and entity['start'] < e['end'])]
            if not overlaps:
                final_entities.append(entity)
            else:
                for overlap in overlaps:
                    overlap_key = tuple(sorted([entity['label'], overlap['label']]))
                    overlap_counts[overlap_key] += 1
    final_entities.sort(key=lambda x: x['start'])

    offset = 0
    text = annotation_data['text']
    for entity in final_entities:
        if entity['label'] in entity_labels:
            start = entity['start'] + offset
            end = entity['end'] + offset
            if only_for_formatting:
                if with_abbr:
                    tag = {'entity-label-1': 'EL1', 'entity-label-2': 'EL2'}[entity['label']]
                else:
                    tag = entity['label']
            else:
                if with_abbr:
                    tag = PPS.abbreviations[entity['label']]
                else:
                    tag = entity['label'].replace(' ', '-')
            start_tag = f"<{tag}>"
            end_tag = f"</{tag}>"
            text = text[:start] + start_tag + text[start:end] + end_tag + text[end:]
            offset += len(start_tag) + len(end_tag)

    return text, dict(overlap_counts)

def format_output_ner(output_type_ner: str,
                      entity_labels: List[str],
                      entities: List[Dict],
                      text: str='',
                      only_for_formatting=False):
    """format_output_ner_entity_list"""
    def get_label_order(entity_labels, entities):
        """get_label_order by PPS.entity_lables"""
        try:
            return entity_labels.index(entities['label'])
        except ValueError:
            return len(entity_labels)
    if not only_for_formatting:
        #Negation entity-label checking
        for entity in entities:
            entity['label'] = 'Negation' if 'Negation' in entity['label'] else entity['label']
        entities = [entity for entity in entities if entity['label'] in entity_labels]
        for entity in entities:
            assert entity['label'] in entity_labels, \
                f"Found undefined entity label in data! - {entity['label']}"
    if 'tagged_text' in output_type_ner:
        assert bool(text), "text must not void when output_type_ner is tagged_text!"

    #sorting annotated entities
    entities = sorted(entities, key=lambda x: (get_label_order(entity_labels, x), x['start']))

    #set formatted_output_ner by output_type_ner
    if output_type_ner=='entity_list_with_start_end':
        formatted_output_ner = ""
        for entity_label in entity_labels:
            formatted_output_ner += f"###{entity_label}###\n"
            for entity in entities:
                text = entity['text']
                start = entity['start']
                end = entity['end']
                entity_label_i = entity['label']
                if entity_label_i == entity_label:
                    formatted_output_ner += f'* "{text}" (start: {start}, end: {end})\n'
            formatted_output_ner += "\n"
    elif output_type_ner=='entity_list_appearance_order_with_entity_label':
        formatted_output_ner = ""
        #re-sorting annotated entities -- only by start position
        entities = sorted(entities, key=lambda x: (x['start']))
        for entity in entities:
            text = entity['text']
            entity_label = entity['label']
            formatted_output_ner += f'* "{text}" (entity label: {entity_label})\n'
        formatted_output_ner += "\n"
    elif output_type_ner=='entity_list_appearance_order_json_format':
        #re-sorting annotated entities -- only by start position
        entities = sorted(entities, key=lambda x: (x['start']))
        entities_print = []
        for entity in entities:
            entities_print.append({'text': entity['text'], 'label': entity['label']})
        formatted_output_ner = str(entities_print) + "\n"
    elif output_type_ner=='tagged_text_with_abbreviations':
        formatted_output_ner, _ = \
            create_tagged_text_with_priority({'text': text, 'entities': entities},
                                             entity_labels=entity_labels,
                                             with_abbr=True,
                                             only_for_formatting=only_for_formatting)
    elif output_type_ner=='tagged_text_without_abbrevaitions':
        formatted_output_ner, _ = \
            create_tagged_text_with_priority({'text': text, 'entities': entities},
                                             entity_labels=entity_labels,
                                             with_abbr=False,
                                             only_for_formatting=only_for_formatting)
    else:
        assert False, f"Undeinfed output_type_ner! {output_type_ner}"

    return formatted_output_ner

def import_train_valid_test_data_by_source(document_source):
    """Set train_valid_test_data_in_a_single_list"""
    if document_source == 'inhouse':
        with open(f"{PPS.dir_train_valid_test_data}/{PPS.fname_train_valid_test_data_inhouse}",
                'rb') as f:
            train_valid_test_data = pickle.load(f)
    else:
        with open(f"{PPS.dir_train_valid_test_data}/{PPS.fname_train_valid_test_data_ds}",
                  'rb') as f:
            train_valid_test_data = pickle.load(f)

    return train_valid_test_data

def import_df_annotated_data():
    """Set train_valid_test_data_in_a_single_list"""
    df_annotated_data = pd.read_csv(f"{PPS.dir_df_annotated}/df_for_annotation.csv")
    return df_annotated_data

def convert_negation_labels_for_check(train_valid_test_data_in_a_single_list):
    """convert_negation_labels_for_check"""
    for data_i in train_valid_test_data_in_a_single_list:
        for entity in data_i['entities']:
            if 'Negation' in entity['label']:
                entity['label'] = 'Negation'
    return train_valid_test_data_in_a_single_list

def get_results_filename(condition, inference_n,
                         temperature=None,
                         repeat_penalty=None,
                         repeat_last_n=None):
    """Generate results filename based on condition"""
    output_formats_dict = {
        'entity_list_with_start_end': 'EL_start_end',
        'entity_list_appearance_order_with_entity_label': 'EL_with_labels',
        'entity_list_appearance_order_json_format': 'EL_json',
        'tagged_text_with_abbreviations': 'TT_w_abbr',
        'tagged_text_without_abbrevaitions': 'TT_wo_abbr'
    }
    output_type_ner = output_formats_dict[condition['output_type_ner']]
    if condition['demonstration_sample_n']==0:
        filename = \
            f"results_{output_type_ner}_none_examplen0_samplen{condition['sample_n']}"
    else:
        if condition['demonstration_sorting_method']=='descending':
            filename = \
                f"results_{output_type_ner}_{condition['demonstration_selelction_method']}_examplen{condition['demonstration_sample_n']}_samplen{condition['sample_n']}"
        else:
            filename = \
                f"results_{output_type_ner}_{condition['demonstration_selelction_method']}_sorting{condition['demonstration_sorting_method']}_examplen{condition['demonstration_sample_n']}_samplen{condition['sample_n']}"
    if inference_n != condition['sample_n']:
        filename += f"_inferencen{inference_n}"
    if condition['usingmaskedtext']:
        filename += '_maskedtext'
    if condition['subgroup_from']:
        filename += f'_subgroupfrom{condition["subgroup_from"]}'
    if condition['subgroup_to']:
        filename += f'_subgroupfrom{condition["subgroup_to"]}'
    if condition['sample_n_from']:
        filename += f'_samplenfrom{condition["sample_n_from"]}'
    if condition['validation']:
        filename += '_validation'
    if condition['entity_group']:
        filename += '_entitygroup'
    if condition['sentence_entity']:
        filename += '_sentenceentity'
    if condition['addptntinfos']:
        filename += f'_addptntinfos{condition["addptntinfos"]}'
    if temperature:
        filename += f'_temp{temperature}'
    if repeat_penalty:
        filename += f'_penalty{repeat_penalty}'
    if repeat_last_n:
        filename += f'_lastn{repeat_last_n}'
    filename += '.pkl'
    filename = filename.replace('samplen422', 'samplenNone_inferencen422')
    filename = filename.replace('samplen221', 'samplenNone_inferencen211')
    return filename

def save_results(results, condition, inference_n,
                 optimal_llm_params,
                 final_input_prompts_date_vers):
    """save_results"""
    filename = get_results_filename(condition, inference_n,
                                    temperature=optimal_llm_params['temperature'],
                                    repeat_penalty=optimal_llm_params['repeat_penalty'],
                                    repeat_last_n=optimal_llm_params['repeat_last_n'])
    full_path = os.path.join(PPS.dir_results_llm, final_input_prompts_date_vers)
    os.makedirs(full_path, exist_ok=True)
    file_path = os.path.join(full_path, filename)
    with open(file_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"Results saved to {file_path}")

def set_conditions_for_evaluation(
    validation=False,
    entity_group=False,
    sentence_entity=False,
    addptntinfos=False,
    addptntinfosn=6,
    usingmaskedtext=False,
    output_types_ner=None,
    demonstration_selection_methods=None,
    demonstration_sorting_methods=None,
    demonstration_sample_ns=None,
    subgroup_category=None,
    samples_n_from=None,
    ):
    """set_conditions_for_evaluation"""
    output_types_ner = output_types_ner or PPS.output_types_ner
    demonstration_selection_methods = \
        demonstration_selection_methods or PPS.demonstration_selection_methods
    demonstration_sorting_methods = \
        demonstration_sorting_methods or PPS.demonstration_sorting_methods
    demonstration_sample_ns = demonstration_sample_ns or PPS.demonstration_sample_ns
    samples_n_from = samples_n_from or [1480]
    if subgroup_category:
        subgroups_from = DOC_INFOS[subgroup_category]
        subgroups_to = DOC_INFOS[subgroup_category]
    else:
        subgroups_from = [None]
        subgroups_to = [None]

    fixed_params = {
        'validation': validation,
        'entity_group': entity_group,
        'sentence_entity': sentence_entity,
        'addptntinfos': addptntinfos,
        'addptntinfosn': addptntinfosn,
        'usingmaskedtext': usingmaskedtext,
        'sample_n': 211 if validation else 422,
        'subgroup_category': subgroup_category,
    }
    conditions = list((
        {**fixed_params,
         'output_type_ner': ot,
         'demonstration_selelction_method': dselem,
         'demonstration_sorting_method': 'descending' if dselem=='Random' or ds==0 else dsortm,
         'demonstration_sample_n': ds,
         'sample_n_from': snf,
         'subgroup_from': subf,
         'subgroup_to': subt}
        for ot, dselem, dsortm, ds, snf, subf, subt in product(
            output_types_ner,
            demonstration_selection_methods,
            demonstration_sorting_methods,
            demonstration_sample_ns,
            samples_n_from,
            subgroups_from,
            subgroups_to,
        )
    ))
    return conditions

def plot_confusion_matrix(cm, classes):
    """plot_confusion_matrix"""
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()

def calculate_class_metrics(true_labels, pred_labels):
    """calculate_class_metrics"""
    report = classification_report(true_labels, pred_labels, output_dict=True)
    labels_in_metrics = list(set(true_labels + pred_labels))
    labels_for_cm = \
        [entity_label for entity_label in PPS.labels_priority_for_confusion_matrix
            if entity_label in labels_in_metrics]
    cm = confusion_matrix(true_labels, pred_labels,
                          labels=labels_for_cm)
    return report, cm, labels_for_cm

def calculate_class_metrics_with_bootstrap(true_labels, pred_labels,
                                           n_iterations=1000, confidence_level=0.95):
    """
    Calculate classification metrics including bootstrap confidence intervals
    
    Args:
        true_labels (list): Ground truth labels
        pred_labels (list): Predicted labels
        n_iterations (int): Number of bootstrap iterations
        confidence_level (float): Confidence level for intervals (default: 0.95)
        
    Returns:
        tuple: (report, cm, labels_for_cm, bootstrap_results)
    """
    # Original metrics calculation
    report = classification_report(true_labels, pred_labels, output_dict=True)
    labels_in_metrics = list(set(true_labels + pred_labels))
    labels_for_cm = [
        entity_label for entity_label in PPS.labels_priority_for_confusion_matrix
        if entity_label in labels_in_metrics
    ]
    cm = confusion_matrix(true_labels, pred_labels, labels=labels_for_cm)

    # Bootstrap calculations
    n_samples = len(true_labels)
    bootstrap_metrics = {
        'precision': {label: [] for label in labels_in_metrics},
        'recall': {label: [] for label in labels_in_metrics},
        'f1-score': {label: [] for label in labels_in_metrics},
        'weighted avg': {'precision': [], 'recall': [], 'f1-score': []},
        'macro avg': {'precision': [], 'recall': [], 'f1-score': []},
        'accuracy': []
    }

    # Perform bootstrap iterations
    for _ in range(n_iterations):
        # Generate bootstrap sample indices
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        # Get bootstrap samples
        bootstrap_true = [true_labels[i] for i in indices]
        bootstrap_pred = [pred_labels[i] for i in indices]
        # Calculate metrics for this bootstrap sample
        bootstrap_report = classification_report(bootstrap_true, bootstrap_pred,
                                              output_dict=True)

        # Store metrics for each label
        for label in labels_in_metrics:
            if label in bootstrap_report:
                bootstrap_metrics['precision'][label].append(
                    bootstrap_report[label]['precision']
                )
                bootstrap_metrics['recall'][label].append(
                    bootstrap_report[label]['recall']
                )
                bootstrap_metrics['f1-score'][label].append(
                    bootstrap_report[label]['f1-score']
                )
        # Store weighted averages
        bootstrap_metrics['weighted avg']['precision'].append(
            bootstrap_report['weighted avg']['precision']
        )
        bootstrap_metrics['weighted avg']['recall'].append(
            bootstrap_report['weighted avg']['recall']
        )
        bootstrap_metrics['weighted avg']['f1-score'].append(
            bootstrap_report['weighted avg']['f1-score']
        )
        # Store macro averages
        bootstrap_metrics['macro avg']['precision'].append(
            bootstrap_report['macro avg']['precision']
        )
        bootstrap_metrics['macro avg']['recall'].append(
            bootstrap_report['macro avg']['recall']
        )
        bootstrap_metrics['macro avg']['f1-score'].append(
            bootstrap_report['macro avg']['f1-score']
        )
        # Store accuracy
        bootstrap_metrics['accuracy'].append(
            bootstrap_report['accuracy']
        )

    # Calculate confidence intervals
    alpha = 1 - confidence_level
    bootstrap_results = {
        'precision': {},
        'recall': {},
        'f1-score': {},
        'weighted avg': {},
        'macro avg': {},
        'accuracy': {}
    }

    # Calculate CIs for individual labels
    for metric, values_by_label in bootstrap_metrics.items():
        if metric in ['precision', 'recall', 'f1-score']:
            for label in labels_in_metrics:
                if label in bootstrap_report:
                    values = values_by_label[label]
                    ci_lower = np.percentile(values, alpha/2 * 100)
                    ci_upper = np.percentile(values, (1 - alpha/2) * 100)
                    bootstrap_results[metric][label] = {
                        'mean': np.mean(values),
                        'ci_lower': ci_lower,
                        'ci_upper': ci_upper
                    }

    # Calculate CIs for weighted and macro averages
    for avg_type in ['weighted avg', 'macro avg']:
        for metric in ['precision', 'recall', 'f1-score']:
            values = bootstrap_metrics[avg_type][metric]
            ci_lower = np.percentile(values, alpha/2 * 100)
            ci_upper = np.percentile(values, (1 - alpha/2) * 100)
            if avg_type not in bootstrap_results:
                bootstrap_results[avg_type] = {}
            bootstrap_results[avg_type][metric] = {
                'mean': np.mean(values),
                'ci_lower': ci_lower,
                'ci_upper': ci_upper
            }

    # Calculate CIs for accuracy
    accuracy_values = bootstrap_metrics['accuracy']
    ci_lower = np.percentile(accuracy_values, alpha/2 * 100)
    ci_upper = np.percentile(accuracy_values, (1 - alpha/2) * 100)
    bootstrap_results['accuracy'] = {
        'mean': np.mean(accuracy_values),
        'ci_lower': ci_lower,
        'ci_upper': ci_upper
    }

    return report, cm, labels_for_cm, bootstrap_results

def format_results_with_ci(report, bootstrap_results):
    """
    Format the results with confidence intervals in a readable way
    
    Args:
        report (dict): Original classification report
        bootstrap_results (dict): Bootstrap confidence intervals
        
    Returns:
        dict: Formatted results with CIs
    """
    formatted_results = {}
    for label in report:
        if isinstance(report[label], dict):  # Skip averaging metrics
            formatted_results[label] = {}
            for metric in ['precision', 'recall', 'f1-score']:
                if label in bootstrap_results[metric]:
                    ci_result = bootstrap_results[metric][label]
                    formatted_results[label][metric] = {
                        'value': report[label][metric],
                        'ci': f"({ci_result['ci_lower']:.3f}, {ci_result['ci_upper']:.3f})"
                    }
                else:
                    formatted_results[label][metric] = {
                        'value': report[label][metric],
                        'ci': 'N/A'
                    }

    return formatted_results

def count_tokens_from_prompt(prompt: str, text: str) -> int:
    """
    Calculates token count from text's position to the end of prompt using LLaMA2 tokenizer
    
    Args:
        prompt (str): Full prompt string
        text (str): Text contained within the prompt
        
    Returns:
        int: Number of tokens. Returns -1 if tokenization fails
    """
    try:
        # Find starting position of text
        text_start_idx = prompt.find(text)
        if text_start_idx == -1:
            raise ValueError("Text not found in prompt")
        # Get substring from text position to end
        target_text = prompt[text_start_idx:]
        # Load tokenizer and count tokens
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
        return len(tokenizer.encode(target_text))

    except Exception as e:
        print(f"Error during tokenization: {str(e)}")
        return -1

def get_condition_from_fname_pkl(fname_pkl):
    """get_condition_from_fname_pkl"""
    # find entity_group
    validation = bool('validation' in fname_pkl)
    # find entity_group
    entity_group = bool('entitygroup' in fname_pkl)
    # find sentence_entity
    sentence_entity = bool('sentenceentity' in fname_pkl)
    # find addptntinfos
    addptntinfos = bool('addptntinfos' in fname_pkl)
    # find usingmaskedtext
    usingmaskedtext = bool('maskedtext' in fname_pkl)
    try:
        if 'addptntinfos' in fname_pkl:
            # Extract the number after 'addptntinfos' and convert to int
            addptntinfosn = int(fname_pkl.split('addptntinfos')[1][0])
        else:
            addptntinfosn = 6
    except (IndexError, ValueError):  # Handle exceptions separately with proper syntax
        addptntinfosn = 6

    #find subgroupcate, subgroupfrom and subgroupto
    subgroupcate = None
    if 'subgroupfrom' in fname_pkl:
        subgroupcate = fname_pkl.split('_subgroupcate')[1].split('_subgroupfrom')[0]
    subgroupfrom = None
    if 'subgroupfrom' in fname_pkl:
        subgroupfrom = fname_pkl.split('subgroupfrom')[1].split('_subgroupto')[0]
        if subgroupfrom in ['2010', '2015', '2020']:
            subgroupfrom = int(subgroupfrom)
    subgroupto = None
    if 'subgroupto' in fname_pkl:
        subgroupto = fname_pkl.split('subgroupto')[1].split('_samplenfrom')[0]
        if subgroupto in ['2010', '2015', '2020']:
            subgroupto = int(subgroupto)
    #find samplenfrom
    samplenfrom = int(fname_pkl.split('samplenfrom')[1].split('.')[0].split('_')[0]) \
        if 'samplenfrom' in fname_pkl else None

    # find output_types_ner
    output_types_ner_found = [otn for otn in PPS.output_types_ner if otn in fname_pkl]
    if len(output_types_ner_found)==1:
        output_types_ner = output_types_ner_found[0]
    elif 'defset_' in fname_pkl:
        output_types_ner = 'entity_list_with_start_end'
    elif len(output_types_ner_found)==0:
        print("No output type found in fname_pkl!")
    else:
        print("More than one output type found in fname_pkl!")
    # find demonstration_selelction_method
    if 'defset_' in fname_pkl:
        demonstration_selelction_method = 'Embedding'
    else:
        demonstration_selelction_method = \
            fname_pkl.split('_sorting')[0].split('_')[1]
    # find demonstration_sorting_method
    if 'defset_' in fname_pkl:
        demonstration_sorting_method = 'descending'
    else:
        demonstration_sorting_method = \
            fname_pkl.split('_examples')[0].split('_')[1].split('_sorting')[1]
    # find demonstration_sample_n
    if 'defset_' in fname_pkl:
        demonstration_sample_n = '15'
    else:
        demonstration_sample_n = int(fname_pkl.split("_examples")[1].split('_')[0])

    sample_n = int(fname_pkl.split("_samplen")[1].split('_')[0].split('.')[0])

    return {'validation': validation,
            'entity_group': entity_group,
            'sentence_entity': sentence_entity,
            'addptntinfos': addptntinfos,
            'addptntinfosn': addptntinfosn,
            'usingmaskedtext': usingmaskedtext,
            'output_type_ner': output_types_ner,
            'demonstration_selelction_method': demonstration_selelction_method,
            'demonstration_sorting_method': demonstration_sorting_method,
            'demonstration_sample_n': demonstration_sample_n,
            'sample_n': sample_n,
            'subgroup_category': subgroupcate,
            'subgroup_from': subgroupfrom,
            'subgroup_to': subgroupto,
            'sample_n_from': samplenfrom}

def select_entities_before_calculate_performance(entities,
                                                 llm_entities,
                                                 sentence_entity,
                                                 entity_group):
    """select_entities_before_calculate_performance"""
    # For sentence_entity
    if sentence_entity:
        entities_selected = []
        llm_entities_selected = []
        for entity in entities:
            if entity['label'] in PPS.sentence_entity_labels:
                entities_selected.append(entity)
        for entity in llm_entities:
            if entity['label'] in PPS.sentence_entity_labels:
                llm_entities_selected.append(entity)
    else:
        entities_selected = []
        llm_entities_selected = []
        for entity in entities:
            if entity['label'] not in PPS.sentence_entity_labels:
                entities_selected.append(entity)
        for entity in llm_entities:
            if entity['label'] not in PPS.sentence_entity_labels:
                llm_entities_selected.append(entity)

    # For entity_group
    if entity_group:
        label2group = \
            {entity_label: group_name
                for group_name, entity_labels in PPS.entity_groups.items()
                for entity_label in entity_labels}
        for entity in entities_selected:
            entity['label'] = label2group.get(entity['label'],
                                              entity['label'])
        for entity in llm_entities_selected:
            entity['label'] = label2group.get(entity['label'],
                                              entity['label'])

    return entities_selected, llm_entities_selected

def set_patient_summaries_for_test_data_i(test_data_i,
                                          df_patient_infos_summaries,
                                          patient_infos_categories):
    """
    Set patient summaries for test data, combining only non-NaN values and formatting them properly.
    
    Args:
        test_data_i: Dictionary containing document_id
        df_patient_infos_summaries: DataFrame containing patient information summaries
    
    Returns:
        str: Formatted patient summaries string
    """
    test_data_index = test_data_i['document_id']
    patient_infos_summaries_i = df_patient_infos_summaries.iloc[test_data_index]
    # Initialize with the English translation of the introduction
    patient_summaries_str = """The following are the patient's medical records prior to note writing for reference. Please use this information when extracting clinical information:\n\n"""
    # Define summaries and their labels
    summary_fields = {
        'summary_basic_with_note': 'Basic Information and Notes',
        'summary_diagnosis': 'Diagnosis History',
        'summary_long_term_prescription': 'Long-term Prescription',
        'summary_short_term_prescription': 'Short-term Prescription',
        'summary_recent_prescription': 'Recent Prescription',
        'summary_lab_results': 'Laboratory Results',
    }
    # Add each non-NaN summary with appropriate formatting
    for field in patient_infos_categories:
        label = summary_fields[field]
        value = patient_infos_summaries_i[field]
        if pd.notna(value):
            patient_summaries_str += f"{label}:\n{value}\n\n"
    return patient_summaries_str.strip()

DIR_SOUBGROUPS = r'E:\NRF_CCADD\DATASET\240705\subgroups'
def set_doc_infos():
    """Set doc_infos by loading from pickle files and validate integer document IDs"""
    fnames_pkl = [fname for fname in os.listdir(DIR_SOUBGROUPS) if fname.endswith('.pkl')]
    doc_infos = {}
    for fname_pkl in fnames_pkl:
        subgroup_category = fname_pkl.split('by_')[1].split('.pkl')[0]
        with open(f"{DIR_SOUBGROUPS}/{fname_pkl}", 'rb') as f:
            document_ids_by = pickle.load(f)

        # Validate that all values in document_ids_by are integers
        for key, value_list in document_ids_by.items():
            if not all(isinstance(doc_id, int) for doc_id in value_list):
                raise ValueError(f"Non-integer document ID found in category '{subgroup_category}', group '{key}'")
        doc_infos[subgroup_category] = document_ids_by

    return doc_infos

DOC_INFOS = set_doc_infos()
DOC_INFOS['diagnosis']['blood_immune_only'] = DOC_INFOS['diagnosis'].pop('blood/immune_only')

def set_n_samples_settings(doc_ids):
    """set n_sample_settings"""
    n_samples_for_valid = 200 if len(doc_ids)>=400 else 50
    n_samples_for_train_max = len(doc_ids) - n_samples_for_valid
    valid_from_other_subgroup = False
    if n_samples_for_train_max < 50:
        n_samples_for_train_max = len(doc_ids)
        valid_from_other_subgroup = True
    return n_samples_for_train_max, n_samples_for_valid, valid_from_other_subgroup

def get_optimal_params(repeat_penalty=None,
                       repeat_last_n=None,
                       temperature=0.001,
                       use_optimal_parmas=True,):
    """get_optimal_params"""
    if use_optimal_parmas:
        #set_llm_optimal_params()
        optimal_llm_params = set_llm_optimal_params()
    else:
        optimal_llm_params = {
            'temperature': temperature,
            'repeat_penalty': repeat_penalty,
            'repeat_last_n': repeat_last_n
        }
    return optimal_llm_params

def add_random_splits_to_doc_infos(doc_infos, start=0, end=2119, parts=3):
    """
    Add random splits of numbers to an existing DOC_INFOS dictionary.
    
    Args:
        doc_infos: Existing dictionary to update
        start: Starting number (inclusive)
        end: Ending number (inclusive)
        parts: Number of parts to split into
    
    Returns:
        Updated dictionary
    """
    # Create and shuffle the numbers
    all_nums = list(range(start, end + 1))
    random.shuffle(all_nums)
    
    # Calculate part size
    part_size = len(all_nums) // parts
    
    # Create the lists
    random_lists = []
    for i in range(parts):
        start_idx = i * part_size
        end_idx = (i+1) * part_size if i < parts-1 else len(all_nums)
        random_lists.append(all_nums[start_idx:end_idx])
    
    # Add to the dictionary
    if 'random' not in doc_infos:
        doc_infos['random'] = {}
    
    for i in range(parts):
        doc_infos['random'][f'random_{i+1}'] = random_lists[i]
    
    return doc_infos

"""Inference and test LLM for ICL biomedical IE"""
import sys
import os
import time

from collections import defaultdict

import gc
import pickle
import requests
from tqdm import tqdm

from transformers import XLMRobertaTokenizerFast

from settings import (
    PPS,
    set_llm_optimal_params
    )
from set_final_prompts import set_final_prompts
from utils import (
    DOC_INFOS,
    set_n_samples_settings,
    set_entity_labels_for_entity_group_sentence,
    get_results_filename,
    save_results,
    set_conditions_for_evaluation,
    calculate_class_metrics_with_bootstrap,
    count_tokens_from_prompt,
    get_condition_from_fname_pkl,
    select_entities_before_calculate_performance,
    get_optimal_params,
    add_random_splits_to_doc_infos
    )
from parsing_outputs import (
    parse_llm_response,
    make_tokens_readable,
    get_entities_in_output_format
)

#set your base directory
DIR_BASE = r'C:\Users\username\Documents\projects\ICLforclinicalIE'
from global_setting import ENTITY_PRIORITY
sys.path.append(rf'{DIR_BASE}\bert_finetuning')
from utils_bertfinetuning import set_subgroups_from_to_and_size_n

DOC_INFOS = add_random_splits_to_doc_infos(DOC_INFOS)
#Global Variables
#need to check ollama API is running in localhost
OLLAMA_API_URL = "http://localhost:11434/api/generate"
TOKENIZER = XLMRobertaTokenizerFast.from_pretrained("FacebookAI/xlm-roberta-large")
LLM_MODEL = 'llama3.3'

FINAL_INPUT_PROMPTS_DATE_VERS = 'prompt_path_for_experiments'
FINAL_RESULTS_DATE_VERS = FINAL_INPUT_PROMPTS_DATE_VERS
DOCUMENT_SOURCE = 'inhouse' # 'inhouse' for SNUH corpus, 'discharge_summary' for MIMIC-III corpus
TEST_IN_DISCHARGE = False #True for test in MIMIC-III discharge summaries, False for SNUH corpus
CONDITION_WITHOUT_SAMPLE_N = True #whether or not use 'sample_n' values in selecting input prompt files befere llm inference/evaluation
VERBOSE = False


# Ollama settings
CONTEXT_LENGTH = 32000  #num_ctx (you can increase this if you have enough memory)
USE_OPTIMAL_PARAMS = True # whether to use optimal LLM parameters or not
TEMPERATURE = 0.001 # temperature for LLM inference

# Additional instructions for json format and tagged text
ADDITIONAL_INSTRUCTION_JSON_FORMAT = """Lastly, the 'label' value must be one of the following:
{entity_labels_joined}. 
Do not use any labels other than those provided in the list above.
"""
ADDITIONAL_INSTRUCTION_TAGGED_TEXT = """Lastly, the tag must be one of the following:
{tag_label_joined}. 
Do not use any tags other than those provided in the list above.
"""

#Functions
def get_llm_response(prompt,
                     temperature=TEMPERATURE,
                     repeat_penalty=None,
                     repeat_last_n=None,
                     token_counts_for_last_n=None):
    """get_llm_response"""
    data = {
        "model": LLM_MODEL,
        "prompt": prompt,
        "stream": False,
   }

    if temperature:
        data['temperature'] = temperature
    if repeat_penalty:
        data['repeat_penalty'] = repeat_penalty
    if repeat_last_n:
        if repeat_last_n == 'text_length':
            assert token_counts_for_last_n is not None, \
                "If repeat_last_n is text_length, token_counts_for_last_n must be not None!"
            data['repeat_last_n'] = token_counts_for_last_n
        else:
            data['repeat_last_n'] = repeat_last_n
    if CONTEXT_LENGTH:
        data['options'] = {}
        data['options']['num_ctx'] = CONTEXT_LENGTH

    response = requests.post(OLLAMA_API_URL, json=data)

    return response.json()['response']

def tokenize_and_label(text, entities):
    """
    Tokenize the input text and label tokens based on given entities.
    Handles multiple occurrences of entities and prevents nested labeling.
    """
    tokens = TOKENIZER.tokenize(text)
    token_labels = ['O'] * len(tokens)
    for entity in sorted(entities, key=lambda x: ENTITY_PRIORITY.get(x['label'], 100)):
        text_ori = entity['text']
        texts = [text_ori, text_ori.lower(), text_ori.capitalize()]
        entity_tokens = TOKENIZER.tokenize(text_ori)
        entity_length = len(entity_tokens)
        for i in range(len(tokens) - entity_length + 1):
            if all(token_labels[j] == 'O' for j in range(i, i + entity_length)):
                string_from_tokens = TOKENIZER.convert_tokens_to_string(tokens[i:i + entity_length])
                if any(string_from_tokens == text for text in texts):
                    for j in range(i, i + entity_length):
                        token_labels[j] = entity['label']
    token_labels = [tl if tl in PPS.entity_labels else 'O' for tl in token_labels]

    return tokens, token_labels

def inference_llm_and_parse_response(prompt,
                                     data_infos_dict,
                                     condition,
                                     optimal_llm_params,
                                     max_trial=5,
                                     printout_i=False):
    """inference_llm_and_parse_response"""
    #set output_type_ner, text and entities
    output_type_ner = condition['output_type_ner']
    entity_group = condition['entity_group']
    sentence_entity = condition['sentence_entity']
    text = data_infos_dict['text']
    entities = data_infos_dict['entities']
    #set entity_labels for entity_group and sentence_entity
    entity_labels = \
        set_entity_labels_for_entity_group_sentence(condition['entity_group'],
                                                    condition['sentence_entity'])

    # Add last instruction for output_type_ner
    if output_type_ner=='entity_list_appearance_order_json_format':
        entity_labels_joined = ', '.join(PPS.entity_labels)
        prompt += '\n' + ADDITIONAL_INSTRUCTION_JSON_FORMAT.format(
            entity_labels_joined=entity_labels_joined
            )
    if output_type_ner=='entity_list_appearance_order':
        tag_label_joined = \
            ', '.join([f"<{PPS.abbreviations[entity_label]}> (label: {entity_label})"
                        for entity_label in entity_labels])
        prompt += '\n' + ADDITIONAL_INSTRUCTION_TAGGED_TEXT.format(
            entity_labels_joined=tag_label_joined
            )
    token_counts_for_last_n = count_tokens_from_prompt(prompt, text)
    optimal_llm_params['token_counts_for_last_n'] = \
        token_counts_for_last_n if token_counts_for_last_n else None

    input_tokens = len(TOKENIZER.encode(prompt))
    llm_responses_failed_to_parse = []

    llm_entities = defaultdict(str)
    parsing_success = False
    for trial in range(max_trial):
        try:
            start_time = time.time()
            if VERBOSE:
                print(f"prompt: {prompt}")
            llm_response = get_llm_response(prompt, **optimal_llm_params)
            if VERBOSE:
                print(f"llm_response: {llm_response}")
            end_time = time.time()
            output_tokens = len(TOKENIZER.encode(llm_response))
            llm_entities = parse_llm_response(llm_response, text,
                                              output_type_ner, entity_labels,
                                              entity_group, sentence_entity)
            parsing_success = True
            break
        except Exception as e:
            print(f"Attempt {trial + 1} failed. Error: {str(e)}")
            llm_responses_failed_to_parse.append(llm_response)
    if not parsing_success:
        print(f"Failed to parse LLM response after {max_trial} attempts.")
        print("Proceeding with empty entities...")
    # Calculate processing time and estimated CO2 emissions
    processing_time = end_time - start_time
    total_tokens = input_tokens + output_tokens
    estimated_co2 = total_tokens * 0.2 / 1000000  # Rough estimate: 0.2g CO2 per 1M tokens

    # select_entities_before_calculate_performance
    entities, llm_entities = \
        select_entities_before_calculate_performance(
            entities=entities,
            llm_entities=llm_entities,
            sentence_entity=condition['sentence_entity'],
            entity_group=condition['entity_group']
            )
    # get true_response
    true_response = get_entities_in_output_format(output_type_ner,
                                                  condition,
                                                  text,
                                                  entities)
    if printout_i:
        print('--------------------------------'*5)
        print('>prompt')
        print(prompt)
        print('--------------------------------'*5)
        print('>original text')
        print(text)
        print('--------------------------------'*5)
        print('>llm_response')
        print(llm_response)
        print('--------------------------------'*5)
        print('>entities extracted by llm')
        print(llm_entities)
        print('--------------------------------'*5)
        print('>entities in output format')
        print(true_response)
        print('--------------------------------'*5)
        print('>annotated entities')
        print([{k: v for k, v in entity.items() if k in ['text', 'start', 'end', 'label']}
                for entity in entities])
        print('++++++++++++++++++++++++++++++++++++++++++++++++++++++')

    # tokenize and labeling
    true_tokens, true_labels = tokenize_and_label(text, entities)
    pred_tokens, pred_labels = tokenize_and_label(text, llm_entities)
    readable_tokens = make_tokens_readable(true_tokens)
    assert len(true_tokens)==len(pred_tokens), \
        "Lengths of true/predicted tokens must be equal!"
    if printout_i:
        print('--------------------------------')
        print('>tokens - true/predicted')
        for rt, _, tl, tp in zip(readable_tokens, true_tokens, true_labels, pred_labels):
            print(rt, tl, tp)
    # set llm_inference_stats
    llm_inference_stats = {
        'input_tokens': input_tokens,
        'output_tokens': output_tokens,
        'processing_time': processing_time,
        'estimated_co2': estimated_co2
        }

    return (prompt, llm_response, llm_entities,
            true_response, entities,
            true_labels, pred_labels, true_tokens,
            llm_inference_stats, llm_responses_failed_to_parse)

def set_fnames_final_input_prompts():
    """set_fnames_final_input_prompts"""
    dir_final_input_prompts = \
        f"{PPS.dir_final_input_prompts}/{FINAL_INPUT_PROMPTS_DATE_VERS}"
    fnames_pkl = \
        [fname for fname
            in os.listdir(dir_final_input_prompts)
            if fname.endswith('.pkl')]
    condition_for_fnames = [get_condition_from_fname_pkl(fname_pkl) for fname_pkl in fnames_pkl]
    return fnames_pkl, condition_for_fnames

def get_final_input_prompts_by_condition(condition,
                                         using_text_with_linebreaks):
    """select_and_read_fnames_pkl_tuples"""
    dir_final_input_prompts = os.path.abspath(
        f"{PPS.dir_final_input_prompts}/{FINAL_INPUT_PROMPTS_DATE_VERS}"
    )
    if using_text_with_linebreaks:
        dir_final_input_prompts += '_textwithlinebreaks'
    fnames_pkl, condition_for_fnames = set_fnames_final_input_prompts()

    # Filter out sample_n if needed
    if CONDITION_WITHOUT_SAMPLE_N:
        condition_for_fnames = [
            {k: v for k, v in cond.items() if k != 'sample_n'}
            for cond in condition_for_fnames
        ]
        condition = {k: v for k, v in condition.items() if k != 'sample_n'}

    # Create list of matching candidates
    matching_candidates = []
    target_sample_n_from = condition.get('sample_n_from', 1480)
    for (fname_pkl, condition_for_fname) in zip(fnames_pkl, condition_for_fnames):
        # Compare all conditions except sample_n_from
        other_conditions_match = \
            all(str(v) == str(condition[k]) for k, v in condition_for_fname.items()
                    if k in condition and k != 'sample_n_from')
        if other_conditions_match:
            curr_sample_n_from = int(condition_for_fname.get('sample_n_from', 1480))
            matching_candidates.append((fname_pkl, curr_sample_n_from))
    if not matching_candidates:
        print("Cannot find corresponding fname_pkl!")
        return None, None 

    # Filter and select appropriate candidate - by target_sample_n_from
    if target_sample_n_from == 1480:
        # If target is 1480 or not specified, use the largest sample_n_from
        selected_fname, selected_n = max(matching_candidates, key=lambda x: x[1])
        if len(matching_candidates) > 1:
            print("Multiple matches found! Selected fname with largest sample_n_from!")
    else:
        # Filter candidates that are <= target_sample_n_from and get the largest
        valid_candidates = [(f, n) for f, n in matching_candidates if n <= target_sample_n_from]
        if not valid_candidates:
            print(f"No candidates found with sample_n_from <= {target_sample_n_from}")
            return None, None
        selected_fname, selected_n = max(valid_candidates, key=lambda x: x[1])
        if len(valid_candidates) > 1:
            print("Multiple matches found. Selected fname with largest valid sample_n_from!")
        if selected_n != target_sample_n_from:
            print(f"Exact match not found. Using closest smaller sample_n_from: {selected_n} (target was {target_sample_n_from})!")

    # Load and return the selected file
    full_path = os.path.join(dir_final_input_prompts, selected_fname)
    with open(full_path, 'rb') as f:
        final_input_prompts = pickle.load(f)
    return final_input_prompts, selected_fname

def run_evaluation_for_conditions(conditions,
                                  optimal_llm_params,
                                  inference_n=100,
                                  printout=False,
                                  printout_every_n=10,
                                  using_text_with_linebreaks=False):
    """run_evaluation for conditions"""
    print(f"Total number of conditions: {len(list(conditions))}")
    for condition in conditions:
        # try:
        print(condition)
        print('\n'.join(f"{k}: {v}" for k, v in condition.items()))
        final_input_prompts, fname_pkl = get_final_input_prompts_by_condition(condition,
                                                                              using_text_with_linebreaks)

        # Check if results already exist
        n_sample_in_filename = 422 if not condition['validation'] else 211
        results_filename = \
            get_results_filename(condition, n_sample_in_filename,
                                    temperature=optimal_llm_params['temperature'],
                                    repeat_penalty=optimal_llm_params['repeat_penalty'],
                                    repeat_last_n=optimal_llm_params['repeat_last_n'])
        full_path = os.path.join(PPS.dir_results_llm,
                                 FINAL_RESULTS_DATE_VERS,
                                 results_filename)

        print(f"results_filename: {results_filename}")
        if os.path.exists(full_path):
            print("Results for condition already exist. Skipping...")
            print(f"condition - {condition}")
            print("="*90)
            continue

        print(f"Processing: {fname_pkl}")
        final_input_prompts = final_input_prompts[:inference_n]
        if final_input_prompts[0][0] is None:
            print("Finished LLM inference - void prompt found!")
            return None

        results = []
        true_labels_total = []
        pred_labels_total = []
        llm_responses_failed_to_parse_total = []
        total_input_tokens = 0
        total_output_tokens = 0
        total_processing_time = 0
        total_estimated_co2 = 0
        for i, (prompt, data_infos_dict) in enumerate(tqdm(final_input_prompts)):
            printout_i = bool(printout and i%printout_every_n==0)
            (prompt, llm_response, llm_entities,
                true_response, entities,
                true_labels, pred_labels, tokens,
                llm_inference_stats, llm_responses_failed_to_parse) = \
                inference_llm_and_parse_response(prompt, data_infos_dict,
                                                    condition, optimal_llm_params,
                                                    printout_i=printout_i)

            true_labels_total.extend(true_labels)
            pred_labels_total.extend(pred_labels)
            llm_responses_failed_to_parse_total.extend(llm_responses_failed_to_parse)
            total_input_tokens += llm_inference_stats['input_tokens']
            total_output_tokens += llm_inference_stats['output_tokens']
            total_processing_time += llm_inference_stats['processing_time']
            total_estimated_co2 += llm_inference_stats['estimated_co2']
            result = {
                'text': data_infos_dict['text'],
                'prompt': prompt,
                'true_response': true_response,
                'entities': entities,
                'llm_response': llm_response,
                'llm_entities': llm_entities,
                'true_labels': true_labels,
                'pred_labels': pred_labels,
                'tokens': tokens,
                'document_id': data_infos_dict['document_id']
            }
            results.append(result)

        class_report, confusion_mat, class_labels, bootstrap_results = \
            calculate_class_metrics_with_bootstrap(true_labels_total, pred_labels_total)
        results_fin = {
            'condition': condition,
            'results': results,
            'llm_inference_stats': {
                'total_input_tokens': total_input_tokens,
                'total_output_tokens': total_output_tokens,
                'total_processing_time': total_processing_time,
                'total_estimated_co2': total_estimated_co2
            },
            'llm_responses_failed_to_parse': llm_responses_failed_to_parse_total,
            'class_report': class_report,
            'confusion_mat': confusion_mat,
            'class_labels': class_labels,
            'bootstrap_results': bootstrap_results
        }
        save_results(results=results_fin,
                     condition=condition,
                     inference_n=inference_n,
                     optimal_llm_params=optimal_llm_params,
                     final_input_prompts_date_vers=FINAL_RESULTS_DATE_VERS)

        del final_input_prompts, results, true_labels_total, pred_labels_total
        del llm_responses_failed_to_parse_total, class_report, confusion_mat
        del class_labels, bootstrap_results, results_fin
        gc.collect()

        # except Exception as e:
        #     print(f"Error processing condition {condition}: {str(e)}")
        #     continue
    print("Finished LLM inference and saved results!")

def main(set_prompts_before_evaluation=False,
         train_n_small_int=None,
         for_subgroup_anlaysis=False,
         using_text_with_linebreaks=False):
    """main"""
    if set_prompts_before_evaluation:
        date_ver = FINAL_INPUT_PROMPTS_DATE_VERS

        if for_subgroup_anlaysis:
            for subgroup_category, doc_infos_in_category in DOC_INFOS.items():
                print(f" - subgroup_category: {subgroup_category}")
                subgroups_from, subgroups_to, size_n = \
                    set_subgroups_from_to_and_size_n(subgroup_category)
                if size_n is None:
                    continue
                for subgroup_from_i in subgroups_from:
                    doc_ids = doc_infos_in_category[subgroup_from_i]
                    print(f"  - subgroup_from_i: {subgroup_from_i}")
                    print(f"   * document_ids length: {len(doc_ids)}")
                    if FINAL_INPUT_PROMPTS_DATE_VERS==\
                        'subgroup_all_samplenfrom50_baseprompt':
                        doc_ids = doc_infos_in_category[subgroup_from_i]
                        print(f"  - subgroup_from_i: {subgroup_from_i}")
                        print(f"   * document_ids length: {len(doc_ids)}")
                        print(f"   * size_n - demo selection pool: {size_n}")
                        if size_n==50:
                            print(f"pass - size_n is 50: {size_n}")
                            continue
                        for subgroup_to_i in subgroups_to:
                            if subgroup_category and subgroup_from_i and subgroup_to_i:
                                DOCUMENT_SOURCE = 'inhouse'
                                samples_n_from = [50]
                                PPS.output_types_ner = ['tagged_text_with_abbreviations']
                                PPS.demonstration_sample_ns = [20]
                                PPS.demonstration_sorting_methods = ["descending"]
                                PPS.demonstration_selection_methods = ["BM25"]
                                set_final_prompts(document_source=DOCUMENT_SOURCE,
                                                    test_sampel_size=200,
                                                    date_ver=date_ver,
                                                    using_masked_text=False,
                                                    add_patient_infos=False,
                                                    subgroup_category=subgroup_category,
                                                    subgroup_from_i=subgroup_from_i,
                                                    subgroup_to_i=subgroup_to_i,
                                                    train_n_small_int=train_n_small_int,
                                                    using_text_with_linebreaks=False,
                                                    test_in_discharge=False,
                                                    samples_n_from=samples_n_from)
                    elif FINAL_INPUT_PROMPTS_DATE_VERS==\
                        'subgroup_all_samplenfrommax_baseprompt':
                        doc_ids = doc_infos_in_category[subgroup_from_i]
                        print(f"  - subgroup_from_i: {subgroup_from_i}")
                        print(f"   * document_ids length: {len(doc_ids)}")
                        print(f"   * size_n - demo selection pool: {size_n}")
                        for subgroup_to_i in subgroups_to:
                            if subgroup_category and subgroup_from_i and subgroup_to_i:
                                DOCUMENT_SOURCE = 'inhouse'
                                samples_n_from = [size_n]
                                PPS.output_types_ner = ['tagged_text_with_abbreviations']
                                PPS.demonstration_sample_ns = [20]
                                PPS.demonstration_sorting_methods = ["descending"]
                                PPS.demonstration_selection_methods = ["BM25"]
                                set_final_prompts(document_source=DOCUMENT_SOURCE,
                                                    test_sampel_size=200,
                                                    date_ver=date_ver,
                                                    using_masked_text=False,
                                                    add_patient_infos=False,
                                                    subgroup_category=subgroup_category,
                                                    subgroup_from_i=subgroup_from_i,
                                                    subgroup_to_i=subgroup_to_i,
                                                    train_n_small_int=train_n_small_int,
                                                    using_text_with_linebreaks=False,
                                                    test_in_discharge=False,
                                                    samples_n_from=samples_n_from)
        else:
            #default settings for llm inference-experiments
            PPS.output_types_ner = ['tagged_text_with_abbreviations']
            PPS.demonstration_sample_ns = [0, 1, 3, 5, 10, 15, 20]
            PPS.demonstration_sorting_methods = ["descending"]
            PPS.demonstration_selection_methods = ["BM25"]
            samples_n_from = [50, 100, 200, 300]

            print(f"FINAL_INPUT_PROMPTS_DATE_VERS: {FINAL_INPUT_PROMPTS_DATE_VERS}")
            if FINAL_INPUT_PROMPTS_DATE_VERS=='discharge_summary_baseprompt':
                samples_n_from = [297]
                PPS.output_types_ner = ["tagged_text_with_abbreviations"]
                PPS.demonstration_selection_methods = ["Embedding"]
                PPS.demonstration_sorting_methods = ["descending"]
                PPS.demonstration_sample_ns = [20]
                set_final_prompts(document_source='discharge_summary',
                                  test_sampel_size=100,
                                  date_ver=date_ver,
                                  using_masked_text=False,
                                  add_patient_infos=False,
                                  train_n_small_int=train_n_small_int,
                                  using_text_with_linebreaks=False,
                                  test_in_discharge=False,
                                  samples_n_from=samples_n_from)
            elif FINAL_INPUT_PROMPTS_DATE_VERS==\
                'from_inhouse_to_discharge_summary_baseprompt':
                samples_n_from = [1480]
                PPS.output_types_ner = ["tagged_text_with_abbreviations"]
                PPS.demonstration_selection_methods = ["BM25", "Embedding"]
                PPS.demonstration_sorting_methods = ["descending"]
                PPS.demonstration_sample_ns = [20]
                set_final_prompts(document_source='inhouse',
                                  test_sampel_size=100,
                                  date_ver=date_ver,
                                  using_masked_text=False,
                                  add_patient_infos=False,
                                  train_n_small_int=train_n_small_int,
                                  using_text_with_linebreaks=False,
                                  test_in_discharge=True,
                                  samples_n_from=samples_n_from)
            elif FINAL_INPUT_PROMPTS_DATE_VERS==\
                'from_discharge_summary_to_inhouse_baseprompt':
                PPS.output_types_ner = ["tagged_text_with_abbreviations"]
                PPS.demonstration_selection_methods = ["BM25", "Embedding"]
                PPS.demonstration_sorting_methods = ["descending"]
                PPS.demonstration_sample_ns = [20]
                set_final_prompts(document_source='discharge_summary',
                                  test_sampel_size=100,
                                  date_ver=date_ver,
                                  using_masked_text=False,
                                  add_patient_infos=False,
                                  train_n_small_int=train_n_small_int,
                                  using_text_with_linebreaks=False,
                                  samples_n_from=samples_n_from,
                                  test_in_discharge=False,
                                  test_in_inhouse=True)
            elif FINAL_INPUT_PROMPTS_DATE_VERS==\
                'inhouse_diversesamplenfrom_baseprompt':
                DOCUMENT_SOURCE = 'inhouse'
                samples_n_from = [50, 100, 200, 300, 500, 900, 1400]
                PPS.output_types_ner = ['tagged_text_with_abbreviations']
                PPS.demonstration_selection_methods = ["BM25", "Random", "TF-IDF"]
                PPS.demonstration_sorting_methods = ["descending"]
                PPS.demonstration_sample_ns = [20]
                set_final_prompts(document_source=DOCUMENT_SOURCE,
                                  test_sampel_size=200,
                                  date_ver=date_ver,
                                  using_masked_text=False,
                                  add_patient_infos=False,
                                  train_n_small_int=train_n_small_int,
                                  using_text_with_linebreaks=False,
                                  test_in_discharge=False,
                                  samples_n_from=samples_n_from)
                #for Embedding - ["clinicalbert", "roberta"]
                DOCUMENT_SOURCE = 'inhouse'
                samples_n_from = [50, 100, 200, 300, 500, 900, 1400]
                PPS.output_types_ner = ['tagged_text_with_abbreviations']
                PPS.demonstration_selection_methods = ["Embedding"]
                PPS.demonstration_sorting_methods = ["descending"]
                PPS.demonstration_sample_ns = [20]
                for bert_model in ["clinicalbert", "roberta"]:
                    if bert_model=="clinicalbert":
                        PPS.embedding_model = "emilyalsentzer/Bio_ClinicalBERT"
                    elif bert_model=="roberta":
                        PPS.embedding_model = "FacebookAI/roberta-large"
                    else:
                        PPS.embedding_model = None
                        print(f"Unknown bert_model: {bert_model}")
                    set_final_prompts(document_source=DOCUMENT_SOURCE,
                                      test_sampel_size=200,
                                      date_ver=date_ver + f'_{bert_model}',
                                      using_masked_text=False,
                                      add_patient_infos=False,
                                      train_n_small_int=train_n_small_int,
                                      using_text_with_linebreaks=False,
                                      test_in_discharge=False,
                                      samples_n_from=samples_n_from)
            elif FINAL_INPUT_PROMPTS_DATE_VERS==\
                "selection_by_domain_baseprompt":
                DOCUMENT_SOURCE = 'inhouse'
                samples_n_from = [50, 100, 200, 300, 500, 900, 1400]
                PPS.output_types_ner = ['tagged_text_with_abbreviations']
                PPS.demonstration_selection_methods = ["Domain"]
                PPS.demonstration_sorting_methods = ["descending"]
                PPS.demonstration_sample_ns = [20]
                set_final_prompts(document_source=DOCUMENT_SOURCE,
                                  test_sampel_size=200,
                                  date_ver=date_ver,
                                  using_masked_text=False,
                                  add_patient_infos=False,
                                  train_n_small_int=train_n_small_int,
                                  using_text_with_linebreaks=False,
                                  test_in_discharge=False,
                                  samples_n_from=samples_n_from)
            elif FINAL_INPUT_PROMPTS_DATE_VERS==\
                "selective_ann_pools":
                DOCUMENT_SOURCE = 'inhouse'
                samples_n_from = [50, 100, 200, 300, 500, 900]
                PPS.output_types_ner = ['tagged_text_with_abbreviations']
                PPS.demonstration_selection_methods = ["BM25"]
                PPS.demonstration_sorting_methods = ["descending"]
                PPS.demonstration_sample_ns = [20]
                set_final_prompts(document_source=DOCUMENT_SOURCE,
                                  test_sampel_size=200,
                                  date_ver=date_ver,
                                  using_masked_text=False,
                                  add_patient_infos=False,
                                  train_n_small_int=train_n_small_int,
                                  using_text_with_linebreaks=False,
                                  test_in_discharge=False,
                                  samples_n_from=samples_n_from,
                                  selective_ann_pool={
                                      "method": 'BM25',
                                      "lambda": [0.0, 0.3, 0.7, 1.0]
                                      })
            elif FINAL_INPUT_PROMPTS_DATE_VERS==\
                "selective_ann_pools_diff_alpha":
                DOCUMENT_SOURCE = 'inhouse'
                samples_n_from = [50, 100, 200, 300, 500, 900]
                PPS.output_types_ner = ['tagged_text_with_abbreviations']
                PPS.demonstration_selection_methods = ["BM25"]
                PPS.demonstration_sorting_methods = ["descending"]
                PPS.demonstration_sample_ns = [20]
                set_final_prompts(document_source=DOCUMENT_SOURCE,
                                  test_sampel_size=200,
                                  date_ver=date_ver,
                                  using_masked_text=False,
                                  add_patient_infos=False,
                                  train_n_small_int=train_n_small_int,
                                  using_text_with_linebreaks=False,
                                  test_in_discharge=False,
                                  samples_n_from=samples_n_from,
                                  selective_ann_pool={
                                      "method": 'BM25',
                                      "lambda": [1.0],
                                      "alpha": [0.0, 0.3, 0.5, 0.7, 1.0]
                                      })
    else:
        optimal_llm_params = get_optimal_params(use_optimal_parmas=USE_OPTIMAL_PARAMS)
        #main
        if for_subgroup_anlaysis:
            #by subgroups
            for subgroup_category in DOC_INFOS:
                #sample_n_from 200
                conditions = set_conditions_for_evaluation(
                    output_types_ner=['entity_list_with_start_end'],
                    demonstration_selection_methods=['Embedding'],
                    demonstration_sorting_methods=['descending'],
                    demonstration_sample_ns=[15],
                    subgroup_category=subgroup_category,
                    samples_n_from=[200])
                run_evaluation_for_conditions(conditions,
                                              optimal_llm_params=optimal_llm_params,
                                              inference_n=422,
                                              printout=False,
                                              printout_every_n=10,
                                              using_text_with_linebreaks=\
                                                  using_text_with_linebreaks)
        else:
            conditions = set_conditions_for_evaluation(
                output_types_ner=[
                    'entity_list_with_start_end',
                    ],
                demonstration_selection_methods=['Embedding'],
                demonstration_sorting_methods=['descending'],
                demonstration_sample_ns=PPS.demonstration_sample_ns
            )
            run_evaluation_for_conditions(conditions,
                                          optimal_llm_params=optimal_llm_params,
                                          inference_n=422,
                                          printout=False,
                                          printout_every_n=10,
                                          using_text_with_linebreaks=\
                                              using_text_with_linebreaks)

if __name__ == "__main__":
    main(
        set_prompts_before_evaluation=True, #False - to run inference without generating prompts
        for_subgroup_anlaysis=False, #True - to run subgroup analysis
        )

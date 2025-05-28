"""Settings for in-context learning for clinical information extraction"""
import os
import pickle

import dataclasses

import pandas as pd

#Functions for global variables
def set_doc_infos(doc_infos):
    """set doc_infos"""
    doc_infos = {
        'S': list(doc_infos[doc_infos['note_type']=='S']['document_id']),
        'O': list(doc_infos[doc_infos['note_type']=='O']['document_id']),
        'A': list(doc_infos[doc_infos['note_type']=='A']['document_id']),
        'P': list(doc_infos[doc_infos['note_type']=='P']['document_id']),
        2010: list(doc_infos[doc_infos['document_creation_year']==2010]['document_id']),
        2015: list(doc_infos[doc_infos['document_creation_year']==2015]['document_id']),
        2020: list(doc_infos[doc_infos['document_creation_year']==2020]['document_id'])
    }
    doc_infos = {k: [int(v_i.split('document_')[1]) for v_i in v] for k, v in doc_infos.items()}
    return doc_infos

@dataclasses.dataclass
class PPS():
    """preprocessing settings"""
    #set directories
    dir_base = r'C:\Users\username\Documents\projects\ICLforclinicalIE'
    dir_train_data = f'{dir_base}/train_data'
    dir_train_valid_test_data = f'{dir_base}/train_valid_test_data'
    dir_prompt_formats = f'{dir_base}/prompt_formats/base_prompts'
    dir_final_input_prompts = f'{dir_base}/final_input_prompts'
    dir_results_llm = f'{dir_base}/results_llm'
    dir_df_annotated = r'E:\NRF_CCADD\DATASET\0426'

    document_sources = ['inhouse', 'discharge_summary']
    document_source = 'inhouse'
    task_types = ['ner', 're']

    #set train_data
    fname_data_inhoues = 'inhouse_notes_entities_relations_negative_included.pkl'
    with open(f'{dir_train_data}/{fname_data_inhoues}', 'rb') as f:
        data_inhouse = pickle.load(f)
    #converting document_id string to int
    for data_i in data_inhouse:
        data_i['document_id'] = int(data_i['document_id'].replace('document_', ''))

    #set df_inhouse_infos and inhouse_docids2infos
    with open(rf'{dir_base}/doc_infos/inhouse_documents_infos_processed.pkl', 'rb') as f:
        df_inhouse_infos = pickle.load(f)
    df_inhouse_infos['document_id'] = [int(document_id.replace('document_', ''))
                                        for document_id in df_inhouse_infos['document_id']]
    inhouse_docids2infos = \
        df_inhouse_infos.set_index('document_id').to_dict(orient='index').items()

    #set df_patient_infos_summaries
    dir_patient_infos = rf'{dir_base}/doc_infos'
    fname_patinet_infos = 'df_patient_information_summaries.csv'
    df_patient_infos_summaries = pd.read_csv(f'{dir_patient_infos}/{fname_patinet_infos}',
                                             encoding='utf-8-sig')
    with open(f"{dir_patient_infos}/inhouse_documents_stratified_info.pkl", 'rb') as f:
        doc_infos = pickle.load(f)
    doc_infos = set_doc_infos(doc_infos)

    #set subgroups for evaluating the robustness of IE pipelines
    subgroups = ['gender', 'note_type', 'document_year', 'age_group']

    #filenames
    fname_train_valid_test_data_inhouse = 'splits_data_7-1-2_without_with_doctor_ids.pkl'
    fname_train_valid_test_data_ds = \
        'train_data_split_mimic_discharge_summary_250108.pkl'

    #entity labels
    entity_labels = [
        'Patient Symptom and Condition',
        'Confirmed Diagnosis',
        'Working Diagnosis',
        'Medication',
        'Procedure',
        'Test and Measurement',
        'Test Result',
        'Time Information',
        ]
    sentence_entity_labels = [
        'Plan (Sentence-level)',
        'Adverse Event (Sentence-level)'
        ]
    entity_groups = {
        'Symptoms and Diagnosis':[
            'Confirmed Diagnosis',
            'Working Diagnosis',
            'Patient Symptom and Condition'
            ],
        'Interventions and Measures': [
            'Procedure',
            'Test and Measurement',
            'Test Result',
            'Medication'
            ]
    }
    abbreviations = {
        'Confirmed Diagnosis': 'CD',
        'Working Diagnosis': 'WD',
        'Patient Symptom and Condition': 'PSC',
        'Medication': 'MED',
        'Procedure': 'PROC',
        'Test and Measurement': 'TM',
        'Test Result': 'TR',
        'Time Information': 'TI',
        'Adverse Event (Sentence-level)': 'AE',
        'Plan (Sentence-level)': 'PLAN',
        'Interventions and Measures': 'Int.Meas.',
        'Symptoms and Diagnosis': 'Symp.Diag.',
    }
    labels_priority_for_confusion_matrix = [
        'O',
        'Symptoms and Diagnosis',
        'Patient Symptom and Condition',
        'Confirmed Diagnosis',
        'Working Diagnosis',
        'Interventions and Measures',
        'Medication',
        'Procedure',
        'Test and Measurement',
        'Test Result',
        'Time Information',
        'Negation',
        'Plan (Sentence-level)',
        'Adverse Event (Sentence-level)',
    ]

    #Demonstration selection
    embedding_model = "sentence-transformers/all-mpnet-base-v2"

    ## final input prompts - setting variables
    #Output types
    output_types_ner = [
        'entity_list_with_start_end',
        'entity_list_appearance_order_with_entity_label',
        'entity_list_appearance_order_json_format',
        'tagged_text_with_abbreviations',
        ]
    #Demonstration selection methods-counts
    demonstration_selection_methods = [
        "Random",
        "Embedding",
        "TF-IDF",
        "BM25",
        "Domain",
        "clinical_bert",
        "roberta"
        ]
    demonstration_sorting_methods = [
        "descending",
        "ascending",
        "bookend",
        "reverse_bookend",
        "alternating",
        "wave",
    ]
    demonstration_sample_ns = [
        0,
        1,
        3,
        5,
        10,
        15,
        20,
        # 25,
    ]
    subgroups_from = [
        'S', 'O', 'A', 'P',
        2010, 2015, 2020
    ]

    #set common_prompt
    if dir_prompt_formats.endswith('base_prompts'):
        prompt_version = 'base_prompt'
    else:
        prompt_version = 'full_prompt'
    fname_common_prompt_ner = f'prompt_ner_common_{prompt_version}.txt'
    with open(f"{dir_prompt_formats}/{fname_common_prompt_ner}", 'r', encoding='utf-8') as f:
        common_prompt_ner = f.read()
    #set entity_descriptions
    entity_descriptions_ner = {}
    fname_entity_description = 'prompt_ner_entity_description_{entity_label}.txt'
    for entity_label in entity_labels:
        fanem_entity_description_i = fname_entity_description.format(entity_label=entity_label)
        with open(f"{dir_prompt_formats}/{fanem_entity_description_i}",
                  'r', encoding='utf-8') as f:
            entity_description_ner = f.read()
        entity_descriptions_ner[entity_label] = entity_description_ner
    #set final_annotation_instruction
    with open(f"{dir_prompt_formats}/prompt_ner_final_annotation_instruction.txt", 'r',
              encoding='utf-8') as f:
        final_annotation_instructions = f.readlines()
    final_annotation_instruction = \
        [fai.replace('\n', '').strip() for fai in final_annotation_instructions]
    with open(f"{dir_prompt_formats}/prompt_ner_final_annotation_instruction_entity_list_appearance_order_json_format.txt", 'r',
              encoding='utf-8') as f:
        final_annotation_instruction_entity_list_json = f.readlines()
    final_annotation_instruction_entity_list_json = \
        [fai.replace('\n', '').strip() for fai in final_annotation_instruction_entity_list_json]
    with open(f"{dir_prompt_formats}/prompt_ner_final_annotation_instruction_tagged_text.txt", 'r',
              encoding='utf-8') as f:
        final_annotation_instruction_tagged_text = f.readlines()
    final_annotation_instruction_tagged_text = \
        [fai.replace('\n', '').strip() for fai in final_annotation_instruction_tagged_text]

    #common prompt elementsa
    common_prompt_elements = {}
    common_prompt_elements['text_type_name'] = "###Text-Type###"
    common_prompt_elements['output_format_name'] = "###Output-Format###"
    common_prompt_elements['task_description'] = "###Task-Description###"
    if dir_prompt_formats.endswith('241213'):
        common_prompt_elements['confusing_cases'] = \
            "###Confusing-Cases###"
    common_prompt_elements['output_example'] = "###Output-Example###"
    common_prompt_elements['final_annotation_instructions'] = \
        "###Final-Annotation-Instructions###"
    common_prompt_elements['selected_demonstration'] = "###Selected_Demonstrations###"
    common_prompt_elements['test_input_text'] = "###Test_Input_Text###"
    if dir_prompt_formats.endswith('241213'):
        common_prompt_elements['patient_information_summaries'] = \
            "###Patient_Information_Summaries###"
    #output formats - ner
    output_formats_ner = {}
    for output_type_ner in output_types_ner:
        if 'entity_list' in output_type_ner:
            output_format_ner = """<Input {document_type_str}>
{input_text}

<Recognized entity list>
{entity_list}
"""
        elif  'tagged_text' in output_type_ner:
            output_format_ner = """<Input {document_type_str}>
{input_text}

<Tagged text>
{tagged_text}
"""
        else:
            assert False, f'{output_type_ner} is not defined output_type_ner!'
        output_formats_ner[output_type_ner] = output_format_ner

#check common_prompt and common_prompt_elements
assert all(cpe in PPS.common_prompt_ner for cpe in PPS.common_prompt_elements.values()), \
    "Check all common_prompt_elements are contained in common_prompt!"
#check subgroups
assert any(subgroup in PPS.df_inhouse_infos.keys() for subgroup in PPS.subgroups), \
    f"subgroups ({PPS.subgroups}) must be one of key of df_inhouse_infos ({PPS.df_inhouse_infos.keys()})"

@dataclasses.dataclass
class TRS():
    """training settings"""
    total_n_inhouse = 2113
    total_n_ds = 397
    document_source = 'inhouse'
    assert document_source in PPS.document_sources, \
        f"document_source must be one of PPS.document_sources ({PPS.document_sources})!"

    #set data_n
    test_data_n = 300 if document_source == 'inhouse' else 200
    train_data_n = 1600 if document_source == 'inhouse' else 100
    valid_data_n = 200 if document_source == 'inhouse' else 97

    cross_validation = False
    cv_fold = 5
    if cross_validation:
        valid_data_n = 0
        train_data_n = total_n_inhouse - test_data_n if document_source == 'inhouse' \
            else total_n_inhouse - test_data_n
    else: #cross_validation is False
        if document_source == 'inhouse':
            assert train_data_n + valid_data_n + test_data_n <= total_n_inhouse, \
                f"Total number of inhouse samples is {total_n_inhouse}! (last updated at 240905: 2113)"
        else:
            assert train_data_n + valid_data_n + test_data_n <= total_n_ds, \
                f"Total number of inhouse samples is {total_n_ds}! (last updated at 240905: 397)"

def set_llm_optimal_params():
    """set_llm_optimal_params"""
    with open(os.path.join('optimal_params_inferencen422_1106.pkl'),
              'rb') as f:
        optimal_params = pickle.load(f)
    return optimal_params

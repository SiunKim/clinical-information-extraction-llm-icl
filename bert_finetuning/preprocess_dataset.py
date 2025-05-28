"""
BERT dataset preprocessing module for NER and RE tasks.
Handles data processing, tokenization, and dataset creation for both tasks using BERT embeddings.
"""
import sys
import pickle
import os
import random
from tqdm import tqdm


import torch
from torch.utils.data import TensorDataset
from transformers import BertModel, BertTokenizerFast, BertForPreTraining
from transformers import XLMRobertaModel, XLMRobertaTokenizerFast
from kaers_bert_config.tokenization_kobert import KoBERTTokenizer

from utils_bertfinetuning import (
    load_dataset,
    save_dataset,
    analyze_label_distribution,
    check_korean_compatibility,
    set_basefilename
)

#set your main directory
sys.path.append(r'C:\Users\username\Documents\projects\ICLforclinicalIE')
from global_setting import ENTITY_PRIORITY
random.seed(42)
TEST_DATA_N = 200

#Functions for global variables
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

def set_annotators_infos():
    with open(f"{DIR_ANNOTATOR_INFOS}/document_ids_by_annotator_241217.pkl", 'rb') as f:
        document_ids_by_annotator = pickle.load(f)
    return document_ids_by_annotator

#Global Variables
DIR_MAIN = r'C:\Users\username\Documents\projects\ICLforclinicalIE' #set your main directory
DIR_DOC_INFORS = rf'{DIR_MAIN}\doc_infos' #directory where document information is stored
DIR_ANNOTATOR_INFOS = rf'{DIR_MAIN}\ann_infos' #directory where annotator information is stored
DIR_DATASET = rf'{DIR_MAIN}\bert_finetuning\dataset'
DIR_TRAIN_VALID_TEST_DATA = rf'{DIR_MAIN}\train_valid_test_data'
DIR_SOUBGROUPS = rf'{DIR_MAIN}\subgroups'
FNAME_TRAIN_VALID_TEST_DATA = 'splits_data_7-1-2_without_with_doctor_ids.pkl'
FNAME_TRAIN_VALID_TEST_DATA_DS = 'train_data_split_mimic_discharge_summary_250108.pkl'
FNAME_DOC_INFOS = 'inhouse_documents_stratified_info.pkl'
DOC_INFOS = set_doc_infos()
DOC_INFOS['diagnosis']['blood_immune_only'] = DOC_INFOS['diagnosis'].pop('blood/immune_only')
DOC_IDS_BY_ANNOTATOR = set_annotators_infos()

# Priority mapping for entity types in NER tasks
ENTITY_PRIORITY = {
    'Confirmed Diagnosis': 1,
    'Working Diagnosis': 2,
    'Patient Symptom and Condition': 3,
    'Medication': 4,
    'Procedure': 5,
    'Test and Measurement': 6,
    'Test Result': 7,
    'Negation: True': 8,
    'Time Information': 9,
    'Adverse Event (Sentence-level)': 10,
    'Plan (Sentence-level)': 11
}

# Special entity categories
SENTENCE_ENTITIES = ['Adverse Event (Sentence-level)', 'Plan (Sentence-level)']
DROPPED_ENTITIES = ['Negation: True']

def assign_label_to_token(token_span, entities, current_label, entity_priority):
    """Assign the most appropriate label to a token based on entity priority.
    
    Args:
        token_span: Tuple of token start and end positions
        entities: List of entity dictionaries
        current_label: Current token label
        entity_priority: Dictionary of entity type priorities
    
    Returns:
        String representing the assigned label
    """
    best_label = current_label
    best_priority = float('inf')

    # Find entity with highest priority overlapping with token
    for entity in entities:
        if entity['start'] <= token_span[0] < entity['end']:
            priority = entity_priority.get(entity['label'], float('inf'))
            if priority < best_priority:
                best_priority = priority
                best_label = entity['label']
    return best_label

def create_tags_and_select_embeddings(token_labels, tokens, embeddings,
                                      use_bio, special_tokens, idx):
    """Create BIO/IO tags and select corresponding embeddings.
    
    Args:
        token_labels: List of token labels
        tokens: List of tokens
        embeddings: Tensor of token embeddings
        use_bio: Boolean for using BIO vs IO scheme
        special_tokens: Set of special tokens to exclude
        idx: Current processing index
    
    Returns:
        Tuple of (tags list, filtered embeddings list)
    """
    if use_bio:
        tags = []
        embeddings_fin = []
        prev_label = 'O'
        # Process each token and create BIO tags
        for label, token, embedding in zip(token_labels, tokens, embeddings):
            if label == 'O':
                tag = 'O'
            elif label != prev_label:
                tag = f'B-{label}'
            else:
                tag = f'I-{label}'
            prev_label = label

            # Skip special tokens
            if token not in special_tokens:
                tags.append(tag)
                embeddings_fin.append(embedding)
                if idx % 500 == 0:
                    print(f"{token} --- {tag}")
    else:
        # Create IO tags
        tags = []
        embeddings_fin = []
        for label, token, embedding in zip(token_labels, tokens, embeddings):
            if token not in special_tokens:
                tags.append('O' if label == 'O' else f'I-{label}')
                embeddings_fin.append(embedding)

    return tags, embeddings_fin

def create_offset_mapping(text, tokenizer, max_length=512, add_special_tokens=True):
    """
    Creates offset mapping for given text and tokenizer
    
    Args:
        text (str): Input text to encode
        tokenizer: Tokenizer object
        max_length (int): Maximum sequence length
        add_special_tokens (bool): Whether to include special tokens like [CLS], [SEP]
    
    Returns:
        torch.Tensor: Tensor of shape (batch_size, sequence_length, 2) containing offset mappings
    """
    tokens = tokenizer.tokenize(text)
    offset_mapping = []
    current_idx = 0
    for token in tokens:
        # Remove ## for subword tokens
        cleaned_token = token.replace('##', '')
        token_len = len(cleaned_token)
        if token.startswith('##'):
            # For subwords, continue from the last position
            offset_mapping.append((current_idx, current_idx + token_len))
        else:
            # For new words, consider space except at the beginning
            if current_idx != 0:
                current_idx += 1
            offset_mapping.append((current_idx, current_idx + token_len))
        current_idx += token_len

    # Handle special tokens if needed
    if add_special_tokens:
        offset_mapping.insert(0, (0, 0))  # [CLS]
        offset_mapping.append((len(text), len(text)))  # [SEP]
    # Add padding offsets if needed
    while len(offset_mapping) < max_length:
        offset_mapping.append((0, 0))
    # Truncate if exceeds max_length
    offset_mapping = offset_mapping[:max_length]

    # Convert to tensor and add batch dimension
    return torch.tensor([offset_mapping], dtype=torch.long)

def encode_with_offset_mapping(text, tokenizer, max_length=512):
    """
    Encodes text and includes offset mapping in the output
    
    Args:
        text (str): Input text to encode
        tokenizer: Tokenizer object
        max_length (int): Maximum sequence length
    
    Returns:
        dict: Encoded outputs including offset mapping
    """
    # Get basic encoding
    encoded = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        padding='max_length',
        truncation=True,
        max_length=max_length,
        return_tensors='pt'
    )
    # Add offset mapping to the encoded output
    encoded['offset_mapping'] = create_offset_mapping(
        text,
        tokenizer,
        max_length=max_length,
        add_special_tokens=True
    )

    return encoded

def preprocess_data(data, tokenizer, model, device,
                    sentence_entities=False,
                    dropped_entities=True,
                    bert_name=None):
    """Preprocess raw data into BERT embeddings and labels.
    
    Args:
        data: List of dictionaries containing text and annotations
        tokenizer: BERT tokenizer instance
        model: BERT model instance
        device: Torch device
        sentence_entities: Boolean to include sentence-level entities
        dropped_entities: Boolean to exclude specified entities
    
    Returns:
        List of processed data dictionaries
    """
    # Filter entity priorities based on configuration
    entity_priority = ENTITY_PRIORITY
    if sentence_entities is False:
        entity_priority = {k: v for k, v in entity_priority.items()
                         if k not in SENTENCE_ENTITIES}
    if dropped_entities:
        entity_priority = {k: v for k, v in entity_priority.items()
                         if k not in DROPPED_ENTITIES}
    print(f"entity_priority: {entity_priority}")

    model.eval()
    processed_data = []
    total_token_labels = 0
    total_re_labels = 0
    # Process each data item
    for idx, item in enumerate(tqdm(data)):
        text = item['text']
        entities = item['entities']
        relations = item['relations']
        # Encode text with BERT
        if 'kaers-bert' in bert_name:
            encoded = encode_with_offset_mapping(text, tokenizer, max_length=512)
        else:
            encoded = tokenizer.encode_plus(text, add_special_tokens=True,
                                            return_offsets_mapping=True,
                                            padding='max_length', truncation=True,
                                            max_length=512, return_tensors='pt')
        token_ids = encoded['input_ids'].squeeze().to(device)
        attention_mask = encoded['attention_mask'].squeeze().to(device)
        token_spans = encoded['offset_mapping'].squeeze()
        tokens = tokenizer.convert_ids_to_tokens(token_ids)
        # Skip invalid tokens
        if not token_spans.numel() or all(span == (0, 0) for span in token_spans):
            print(f"Warning: No valid tokens for text: {text[:50]}...")
            continue

        # Get BERT embeddings
        with torch.no_grad():
            outputs = model(token_ids.unsqueeze(0),
                            attention_mask=attention_mask.unsqueeze(0))
            if 'kaers-bert' in bert_name:
                embeddings = outputs.prediction_logits.squeeze(0)
            else:
                embeddings = outputs.last_hidden_state.squeeze(0)

        # Create token labels
        token_labels = ['O'] * len(token_spans)
        for i, span in enumerate(token_spans):
            if span != (0, 0):  # Skip padding tokens
                token_labels[i] = assign_label_to_token(span, entities,
                                                      token_labels[i],
                                                      entity_priority)

        # Process relations
        re_labels = []
        for relation in relations:
            try:
                # Find corresponding entities
                from_entity = next(e for e in entities
                                 if e['id'] == relation['from_id'])
                to_entity = next(e for e in entities
                               if e['id'] == relation['to_id'])
                # Skip invalid positions
                if from_entity['start'] > len(text) or to_entity['start'] > len(text):
                    print(f"Warning: Entity position out of bounds for text: {text[:50]}...")
                    continue

                # Find token indices
                from_start = min(range(len(token_spans)),
                               key=lambda i: abs(token_spans[i][0] -
                                                 from_entity['start']))
                to_start = min(range(len(token_spans)),
                             key=lambda i: abs(token_spans[i][0] -
                                               to_entity['start']))
                re_labels.append({
                    'from_token_idx': from_start,
                    'to_token_idx': to_start,
                    'label': relation.get('relation_type', True)
                })
            except StopIteration:
                print(f"Warning: Entity not found for relation in text: {text[:50]}...")
                continue

        # Store processed item
        processed_data.append({
            'tokens': tokens,
            'token_ids': token_ids.cpu(),
            'token_labels': token_labels,
            're_labels': re_labels,
            'embeddings': embeddings.cpu()
        })

        total_token_labels += len(token_labels)
        total_re_labels += len(re_labels)

        # Progress logging
        if (idx + 1) % 500 == 0:
            print(f"Processed {idx + 1} items. "
                  f"Total token labels: {total_token_labels}, "
                  f"Total RE labels: {total_re_labels}")

    return processed_data

def create_ner_re_datasets(processed_data, use_bio, special_tokens,
                           tag_to_index=None,
                           subgroup_from=None):
    """Create NER and RE datasets from processed data.
    
    Args:
        processed_data: List of processed data dictionaries
        use_bio: Boolean for using BIO vs IO scheme
        special_tokens: Set of special tokens to exclude
        tag_to_index: Optional dictionary mapping tags to indices
    
    Returns:
        Tuple of (NER dataset, RE dataset, tag_to_index)
    """
    
    # Create tag_to_index if not provided
    if tag_to_index is None:
        unique_labels = set()
        for item in processed_data:
            unique_labels.update(item['token_labels'])
        tag_to_index = {'O': 0}
        current_index = 1
        # Assign indices to tags
        for label in sorted(unique_labels):
            if label != 'O':
                tag_to_index[f'B-{label}' if use_bio else f'I-{label}'] = current_index
                if use_bio:
                    tag_to_index[f'I-{label}'] = current_index + 1
                    current_index += 2
                else:
                    current_index += 1
        print(f"tag_to_index: {tag_to_index}")

    # Create datasets
    ner_embeddings, ner_labels = [], []
    re_embeddings, re_labels = [], []
    for idx, item in enumerate(processed_data):
        # Process NER data
        tags, embeddings_fin = create_tags_and_select_embeddings(
            item['token_labels'],
            item['tokens'],
            item['embeddings'],
            use_bio,
            special_tokens,
            idx
        )
        ner_embeddings.extend(embeddings_fin)
        ner_labels.extend([tag_to_index[tag] for tag in tags])
        # Process RE data
        for re_label in item['re_labels']:
            from_emb = item['embeddings'][re_label['from_token_idx']]
            to_emb = item['embeddings'][re_label['to_token_idx']]
            re_embeddings.append(torch.cat([from_emb, to_emb]))
            re_labels.append(1 if re_label['label'] is True else 0)

    # Convert to tensors
    ner_embeddings = torch.stack(ner_embeddings)
    ner_labels = torch.tensor(ner_labels, dtype=torch.long)
    re_embeddings = torch.stack(re_embeddings)
    re_labels = torch.tensor(re_labels, dtype=torch.long)

    # Create TensorDatasets
    ner_dataset = TensorDataset(ner_embeddings, ner_labels)
    re_dataset = TensorDataset(re_embeddings, re_labels)

    return ner_dataset, re_dataset, tag_to_index

def process_test_data_by_subgroup_i(test_data,
                                    subgroup_category,
                                    subgroup_to_i,
                                    tokenizer,
                                    model,
                                    device,
                                    sentence_entities,
                                    dropped_entities,
                                    bert_name,
                                    use_bio,
                                    special_tokens,
                                    tag_to_index):
    """process_test_data_by_subgroup_i"""
    print(f"     - subgroup_to_i: {subgroup_to_i}")
    if subgroup_to_i=='total':
        test_data_i = test_data
    else:
        test_data_i = [td for td in test_data
                        if td['document_id'] in DOC_INFOS[subgroup_category][subgroup_to_i]]

    #handle void dataset
    if len(test_data_i)==0:
        return None, None

    test_data_i = random.sample(test_data_i, min(len(test_data_i), TEST_DATA_N))
    test_processed = preprocess_data(
        test_data_i, tokenizer, model, device,
        sentence_entities=sentence_entities,
        dropped_entities=dropped_entities,
        bert_name=bert_name
    )
    test_dataset_ner, test_dataset_re, _ = create_ner_re_datasets(
        test_processed, use_bio, special_tokens, tag_to_index)

    return test_dataset_ner, test_dataset_re

def preprocessing_bert_dataset(bert_name='google-bert/bert-base-uncased',
                               n_samples_for_train=1480,
                               n_samples_for_valid=200,
                               use_bio=True,
                               saving_dataset=False,
                               sentence_entities=False,
                               dropped_entities=True,
                               small_dataset=False,
                               subgroup_category=None,
                               subgroup_from_i=None,
                               subgroups_to=None,
                               valid_from_other_subgroup=False,
                               test_in_best_two_annotators=False,
                               discharge_summary=False):
    """Main function to preprocess BERT dataset for NER and RE tasks.
    
    Args:
        bert_name: Name of BERT model to use
        train/valid/test: Split ratios
        use_bio: Boolean for using BIO vs IO scheme
        saving_dataset: Boolean to save processed datasets
        sentence_entities: Boolean to include sentence-level entities
        dropped_entities: Boolean to exclude specified entities
        small_dataset: Boolean to use reduced dataset size
    
    Returns:
        Tuple of (train_ner, train_re, valid_ner, valid_re, 
                 test_ner, test_re, tag_to_index)
    """
    # Setup device and model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    xlm = 'xlm' in bert_name
    # Initialize tokenizer and model based on type
    if xlm:
        tokenizer = XLMRobertaTokenizerFast.from_pretrained(bert_name)
        model = XLMRobertaModel.from_pretrained(bert_name).to(device)
    else:
        if 'kaers-bert' in bert_name:
            tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')
            model = BertForPreTraining.from_pretrained(bert_name).to(device)
        else:
            tokenizer = BertTokenizerFast.from_pretrained(bert_name)
            model = BertModel.from_pretrained(bert_name).to(device)
    special_tokens = set(tokenizer.all_special_tokens)
    print(f"Special tokens: {special_tokens}")

    # Check Korean compatibility
    korean_compatible = check_korean_compatibility(tokenizer, model, device, bert_name)
    if not korean_compatible:
        print("Consider using a multilingual or Korean-specific BERT model.")

    #check n_samples_for_train
    # Load raw data and split
    if discharge_summary:
        with open(f'{DIR_TRAIN_VALID_TEST_DATA}/{FNAME_TRAIN_VALID_TEST_DATA_DS}', 'rb') as f:
            data_splits = pickle.load(f)
    else:
        with open(f'{DIR_TRAIN_VALID_TEST_DATA}/{FNAME_TRAIN_VALID_TEST_DATA}', 'rb') as f:
            data_splits = pickle.load(f)
    
    # Split data according to specified ratios
    train_data, valid_data, test_data = data_splits
    if test_in_best_two_annotators:
        document_ids_best_two_annotators = \
            [int(document_id.split('_')[1])
                for annotator in ['jhakim@snu.ac.kr', 'nataliejh@naver.com']
                for document_id in DOC_IDS_BY_ANNOTATOR[annotator]]
        test_data = [td for td in test_data
                        if td['document_id'] in document_ids_best_two_annotators]
    if subgroup_category:
        total_data = train_data + valid_data + test_data
        #set document_ids
        document_ids_for_subgroup = DOC_INFOS[subgroup_category][subgroup_from_i]
        document_ids_for_subgroup_test = []
        for subgroup_i, document_ids in DOC_INFOS[subgroup_category].items():
            if subgroup_i!=subgroup_from_i:
                document_ids_for_subgroup_test += document_ids
        #set train_data, valid_data, test_data
        train_valid_data = [td for td in total_data
                                if td['document_id'] in document_ids_for_subgroup]
        if valid_from_other_subgroup:
            print("Set valid_from_other_subgroup!")
            document_ids_in_subgroup_category = \
                [v for vs in DOC_INFOS[subgroup_category].values() for v in vs]
            valid_data = [td for td in total_data
                            if td['document_id'] not in document_ids_in_subgroup_category]
            assert len(valid_data)>50, \
                f"valid_from_other_subgroup {valid_from_other_subgroup} - subgroup_from_i {subgroup_from_i}!"
            valid_data = random.sample(valid_data, 50)
            train_data = train_valid_data[:n_samples_for_train]
        else:
            valid_data = train_valid_data[-n_samples_for_valid:]
            train_data = train_valid_data[:-n_samples_for_valid][:n_samples_for_train]
        #set test_data
        used_doc_ids = set(td['document_id'] for td in train_data + valid_data)
        remaining_data = [td for td in train_valid_data
                            if td['document_id'] not in used_doc_ids]
        test_data = [td for td in total_data
                                if td['document_id'] in document_ids_for_subgroup_test]
        if test_in_best_two_annotators:
            test_data = [td for td in test_data
                            if td['document_id'] in document_ids_best_two_annotators]
        test_data.extend(remaining_data)
    #update n_samples_for_train
    n_samples_for_train = len(train_data)
    n_samples_for_valid = len(valid_data)

    # Setup file paths
    base_filename, bert_subdir = set_basefilename(bert_name, n_samples_for_train, use_bio)
    # Define paths for saved files
    if test_in_best_two_annotators:
        tag_to_index_file = os.path.join(DIR_DATASET, bert_subdir,
                                         'test_in_best_two_annotators',
                                         f"{base_filename}_tag_to_index.pkl")
        train_ner_file = os.path.join(DIR_DATASET, bert_subdir,
                                      'test_in_best_two_annotators',
                                      f"{base_filename}_train_ner.pkl")
    else:
        tag_to_index_file = os.path.join(DIR_DATASET, bert_subdir,
                                         f"{base_filename}_tag_to_index.pkl")
        train_ner_file = os.path.join(DIR_DATASET, bert_subdir,
                                      f"{base_filename}_train_ner.pkl")
    if subgroup_from_i:
        train_ner_file = \
            train_ner_file.replace(".pkl",
                                   f"_subgroupfrom{subgroup_category}{subgroup_from_i}.pkl")
    if discharge_summary:
        tag_to_index_file = tag_to_index_file.replace('.pkl', '_dischargesummary.pkl')
        train_ner_file = train_ner_file.replace('.pkl', '_dischargesummary.pkl')

    # Check if preprocessed files exist and load them
    if os.path.exists(tag_to_index_file) and os.path.exists(train_ner_file):
        print("Loading preprocessed datasets...")
        with open(tag_to_index_file, 'rb') as f:
            tag_to_index = pickle.load(f)
        # Load all datasets
        train_dataset_ner = load_dataset(train_ner_file)
        train_dataset_re = load_dataset(train_ner_file.replace('_ner', '_re'))
        valid_dataset_ner = load_dataset(train_ner_file.replace('_train_', '_valid_'))
        valid_dataset_re = load_dataset(train_ner_file.replace('_train_ner', '_valid_re'))
        test_dataset_ner_by_subgroups = \
            load_dataset(train_ner_file.split('_train_ner_')[0] + "_test_ner_bysubgroups.pkl")
        test_dataset_re_by_subgroups = \
            load_dataset(train_ner_file.split('_train_ner_')[0] + "_test_re_bysubgroups.pkl")
    else:
        print("Preprocessing new datasets...")
        # Handle small dataset option
        if small_dataset:
            train_data = train_data[:100]
            valid_data = valid_data[:100]
            test_data = test_data[:100]

        # Process each dataset split - train_valid
        train_processed = preprocess_data(
            train_data, tokenizer, model, device,
            sentence_entities=sentence_entities,
            dropped_entities=dropped_entities,
            bert_name=bert_name
        )
        valid_processed = preprocess_data(
            valid_data, tokenizer, model, device,
            sentence_entities=sentence_entities,
            dropped_entities=dropped_entities,
            bert_name=bert_name
        )
        # Create datasets
        train_dataset_ner, train_dataset_re, tag_to_index = create_ner_re_datasets(
            train_processed, use_bio, special_tokens)
        valid_dataset_ner, valid_dataset_re, _ = create_ner_re_datasets(
            valid_processed, use_bio, special_tokens, tag_to_index)

        #processing test_data by subgroup_i
        test_dataset_ner_by_subgroups = {}
        test_dataset_re_by_subgroups = {}
        if subgroup_category is None:
            test_processed = preprocess_data(
                test_data, tokenizer, model, device,
                sentence_entities=sentence_entities,
                dropped_entities=dropped_entities,
                bert_name=bert_name
            )
            test_dataset_ner, test_dataset_re, _ = create_ner_re_datasets(
                test_processed, use_bio, special_tokens, tag_to_index)
            test_dataset_ner_by_subgroups['total'] = test_dataset_ner
            test_dataset_re_by_subgroups['total'] = test_dataset_re
        else:
            for subgroup_to_i in subgroups_to:
                test_dataset_ner, test_dataset_re = \
                        process_test_data_by_subgroup_i(test_data=test_data,
                                                        subgroup_category=subgroup_category,
                                                        subgroup_to_i=subgroup_to_i,
                                                        tokenizer=tokenizer,
                                                        model=model,
                                                        device=device,
                                                        sentence_entities=sentence_entities,
                                                        dropped_entities=dropped_entities,
                                                        bert_name=bert_name,
                                                        use_bio=use_bio,
                                                        special_tokens=special_tokens,
                                                        tag_to_index=tag_to_index)
                test_dataset_ner_by_subgroups[subgroup_to_i] = \
                    test_dataset_ner
                test_dataset_re_by_subgroups[subgroup_to_i] = \
                    test_dataset_re

        # Save processed datasets if requested
        if saving_dataset:
            print("Saving preprocessed datasets...")
            os.makedirs(os.path.join(DIR_DATASET, bert_subdir), exist_ok=True)
            # Save tag_to_index
            with open(tag_to_index_file, 'wb') as f:
                pickle.dump(tag_to_index, f)
            # Save all datasets
            save_dataset(train_dataset_ner, train_ner_file)
            save_dataset(train_dataset_re, train_ner_file.replace('_ner', '_re'))
            save_dataset(valid_dataset_ner, train_ner_file.replace('_train_', '_valid_'))
            save_dataset(valid_dataset_re, train_ner_file.replace('_train_ner', '_valid_re'))
            save_dataset(test_dataset_ner_by_subgroups,
                            train_ner_file.split('_train_ner_')[0] + \
                                "_test_ner_bysubgroups.pkl")
            save_dataset(test_dataset_re_by_subgroups,
                            train_ner_file.split('_train_ner_')[0] + \
                                "_test_re_bysubgroups.pkl")

    # Analyze dataset distributions
    datasets = [
        (train_dataset_ner, train_dataset_re),
        (valid_dataset_ner, valid_dataset_re),
        (test_dataset_ner_by_subgroups, test_dataset_re_by_subgroups)
    ]
    dataset_names = ['Train', 'Validation', 'Test']
    analyze_label_distribution(datasets, dataset_names, tag_to_index)

    return (train_dataset_ner, train_dataset_re,
            valid_dataset_ner, valid_dataset_re,
            test_dataset_ner_by_subgroups, test_dataset_re_by_subgroups,
            tag_to_index)

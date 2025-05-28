"""
Data Processing and Model Utilities

This module provides utilities for handling dataset operations and model compatibility checks,
particularly focused on Korean language processing tasks. It includes functions for data splitting,
dataset saving/loading, and BERT model validation for Korean text processing.

Main Features:
    - Data splitting into train/validation/test sets
    - Large dataset handling with chunked saving and loading
    - Korean language compatibility testing for BERT models

Functions:
    split_data(train, valid, test, data_splits):
        Splits data into train, validation, and test sets with configurable ratios.
    
    save_dataset(dataset, filename, chunk_size=100000):
        Saves large datasets to files with optional chunking for memory efficiency.
    
    load_dataset(filename):
        Loads datasets from single or multiple chunked files.
    
    check_korean_compatibility(tokenizer, model, device, bert_name):
        Validates BERT model and tokenizer compatibility with Korean text.

Dependencies:
    - pickle: For dataset serialization
    - os: For file operations
    - torch: For tensor operations and model handling

Note:
    The data splitting function assumes a 10-split structure where training data
    must contain at least 5 splits. The save/load functions handle large datasets
    by automatically chunking them when necessary.
"""
import pickle
import os
from collections import Counter

import matplotlib.pyplot as plt

import pandas as pd
import numpy as np

import torch

from sklearn.metrics import (f1_score, accuracy_score,
                             precision_score, recall_score, roc_auc_score)



def set_basefilename(bert_name, n_samples_for_train, use_bio):
    """create_save_directory"""
    bert_subdir = bert_name.replace('/', '-')
    split_ratio = f"ntrainsamples{n_samples_for_train}"
    tag_type = 'bio' if use_bio else 'io'
    base_filename = f"{split_ratio}_{tag_type}"
    return base_filename, bert_subdir

def save_dataset(dataset, filename):
    """
    Save dataset to file(s) using pickle.
    """
    if not dataset:
        print(f"Warning: Dataset is empty. Not saving {filename}")
        return

    os.makedirs(os.path.dirname(filename), exist_ok=True)
    try:
        with open(filename, 'wb') as f:
            pickle.dump(dataset, f)
        print(f"Saved entire dataset to {filename}")
        if os.path.getsize(filename) == 0:
            print(f"Warning: {filename} is empty after saving")
    except Exception as e:
        print(f"Error saving {filename}: {str(e)}")

    print(f"Total dataset size: {len(dataset)}")

def load_dataset(filename):
    """
    Load dataset from file(s). If multiple chunk files exist,
    it will load and combine them.
    """
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)
    else:
        dataset = []
        chunk = 0
        while True:
            chunk_filename = f"{filename}_chunk_{chunk}.pkl"
            if os.path.exists(chunk_filename):
                with open(chunk_filename, 'rb') as f:
                    dataset.extend(pickle.load(f))
                chunk += 1
            else:
                break
        return dataset

def check_korean_compatibility(tokenizer, model, device, bert_name):
    """
    Check if the given BERT model and tokenizer can properly handle Korean text.
    """
    # Test Korean text
    korean_text = \
        "안녕하세요. 이것은 한국어 테스트입니다. 적절히 한국어를 토크나이징할 수 있는지 확인 중입니다."
    # Tokenize
    encoded = tokenizer.encode_plus(korean_text, add_special_tokens=True, return_tensors='pt')
    input_ids = encoded['input_ids'].to(device)
    attention_mask = encoded['attention_mask'].to(device)
    # Print tokenized result
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    print(f"Tokenized result: {tokens}")
    # Get BERT embeddings
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        if 'kaers-bert' in bert_name:
            embeddings = outputs.prediction_logits
        else:
            embeddings = outputs.last_hidden_state
    # Check if tokenization and embedding generation worked
    if embeddings.shape[1] > 2:  # More than just [CLS] and [SEP] tokens
        print(f"The BERT model '{bert_name}' can handle Korean text.")
        return True
    print(f"Warning: The BERT model '{bert_name}' may not properly handle Korean text.")
    return False

def analyze_label_distribution(datasets, dataset_names, tag_to_index):
    """analyze_label_distribution"""
    def analyze_label_distribution_i(ner_dataset_i, re_dataset_i):
        """analyze_label_distribution_i"""
        print(f"\n{name} Dataset:")
        ner_labels = ner_dataset_i.tensors[1].numpy()
        ner_distribution = Counter(ner_labels)
        total_ner = sum(ner_distribution.values())

        print("  NER Label Distribution:")
        for label, count in sorted(ner_distribution.items()):
            tag = index_to_tag.get(label, f"Unknown-{label}")
            percentage = (count / total_ner) * 100
            print(f"    {tag}: {count} ({percentage:.2f}%)")

        re_labels = re_dataset_i.tensors[1].numpy()
        re_distribution = Counter(re_labels)
        total_re = sum(re_distribution.values())

        print("\n  RE Label Distribution:")
        for label, count in sorted(re_distribution.items()):
            percentage = (count / total_re) * 100
            print(f"    {label}: {count} ({percentage:.2f}%)")
    # Analyze and print the label distribution for NER and RE tasks across multiple datasets
    index_to_tag = {v: k for k, v in tag_to_index.items()}

    for (ner_dataset, re_dataset), name in zip(datasets, dataset_names):
        if name == 'Test':
            for subgroup_i in ner_dataset:
                ner_dataset_i = ner_dataset[subgroup_i]
                re_dataset_i = re_dataset[subgroup_i]
                print(f"Subgroup - {subgroup_i}")
                if ner_dataset_i and re_dataset_i:
                    analyze_label_distribution_i(ner_dataset_i, re_dataset_i)
        else:
            analyze_label_distribution_i(ner_dataset, re_dataset)

def save_results(metrics, file_path):
    """save_results"""
    print(f"metrics: {metrics}")
    with open(file_path, 'w', encoding='utf-8-sig') as f:
        for key, value in metrics.items():
            if isinstance(value, dict):
                f.write(f"{key}:\n")
                for sub_key, sub_value in value.items():
                    f.write(f"  {sub_key}: {sub_value}\n")
            else:
                f.write(f"{key}: {value}\n")

def save_results_excel(metrics, file_path):
    """
    Save metrics to an Excel file.
    
    :param metrics: Dictionary containing metric data
    :param file_path: Path where the Excel file will be saved
    """
    rows = []
    for key, value in metrics.items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                rows.append({'Metric': f"{key}_{sub_key}", 'Value': sub_value})
        else:
            rows.append({'Metric': key, 'Value': value})
    df = pd.DataFrame(rows)
    df.to_excel(file_path, index=False)

def plot_losses(ner_train_losses, ner_val_losses, re_train_losses,
                re_val_losses, hyperparams, save_dir):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(ner_train_losses, label='Train Loss')
    plt.plot(ner_val_losses, label='Validation Loss')
    plt.title(f'NER Model Losses\n(batch_size={hyperparams["batch_size"]}, lr={hyperparams["lr"]}, hidden_dim={hyperparams["hidden_dim"]})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(re_train_losses, label='Train Loss')
    plt.plot(re_val_losses, label='Validation Loss')
    plt.title(f'RE Model Losses\n(batch_size={hyperparams["batch_size"]}, lr={hyperparams["lr"]}, hidden_dim={hyperparams["hidden_dim"]})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    fname_fig = \
        f'loss_plot_bs{hyperparams["batch_size"]}_lr{hyperparams["lr"]}_hd{hyperparams["hidden_dim"]}_do{hyperparams["dropout_rate"]}_wd{hyperparams["weight_decay"]}.png'
    plt.savefig(save_dir / fname_fig)
    plt.close()

def print_metrics(ner_metrics, re_metrics):
    """print_metrics"""
    if ner_metrics:
        print("NER Metrics:")
        print(f"  Accuracy: {ner_metrics['accuracy']:.4f}")
        print(f"  F1 Macro: {ner_metrics['f1_macro']:.4f}")
        print(f"  F1 Micro: {ner_metrics['f1_micro']:.4f}")
        print("  Entity Type Metrics:")
        for entity_type, entity_metrics in ner_metrics['entity_metrics'].items():
            print(f"    {entity_type}:")
            print(f"      Count: {entity_metrics['count']}")
            print(f"      Accuracy: {entity_metrics['accuracy']:.4f}")
            print(f"      Precision: {entity_metrics['precision']:.4f}")
            print(f"      Recall: {entity_metrics['recall']:.4f}")
            print(f"      F1 Score: {entity_metrics['f1']:.4f}")

    if re_metrics:
        print("RE Metrics:")
        print(f"  Accuracy: {re_metrics['accuracy']:.4f}")
        print(f"  F1 Score: {re_metrics['f1']:.4f}")
        print(f"  Precision: {re_metrics['precision']:.4f}")
        print(f"  Recall: {re_metrics['recall']:.4f}")
        print(f"  AUROC: {re_metrics['auroc']:.4f}")

def convert_bio_to_io(tag_sequence, index_to_tag, tag_to_index):
    """Convert B-XXX to I-XXX in tag sequence."""
    new_sequence = []
    for label_id in tag_sequence:
        tag = index_to_tag[label_id]
        if tag.startswith("B-"):
            tag = "I-" + tag[2:]
        new_sequence.append(tag_to_index.get(tag, label_id))  # fallback to original if not found
    return new_sequence

def calculate_ner_metrics(all_preds, all_labels,
                          tag_to_index, bootstrap=False, n_iterations=100):
    """
    Calculate NER metrics with optional bootstrap confidence intervals
    
    Args:
        all_preds: Predicted labels
        all_labels: True labels
        tag_to_index: Dictionary mapping entity types to indices
        bootstrap: Whether to calculate bootstrap CIs
        n_iterations: Number of bootstrap iterations
    """
    base_metrics = {
        'accuracy': accuracy_score(all_labels, all_preds),
        'f1_macro': f1_score(all_labels, all_preds, average='macro'),
        'f1_micro': f1_score(all_labels, all_preds, average='micro'),
        'entity_metrics': {}
    }

    index_to_tag = {v: k for k, v in tag_to_index.items()}
    all_preds = convert_bio_to_io(all_preds, index_to_tag, tag_to_index)
    all_labels = convert_bio_to_io(all_labels, index_to_tag, tag_to_index)

    # Initialize entity_metrics
    for entity_type in tag_to_index:
        base_metrics['entity_metrics'][entity_type] = {'count': 0, 'correct': 0}
    # Calculate base entity metrics
    for true_label, pred_label in zip(all_labels, all_preds):
        entity_type = index_to_tag[true_label]
        base_metrics['entity_metrics'][entity_type]['count'] += 1
        if true_label == pred_label:
            base_metrics['entity_metrics'][entity_type]['correct'] += 1
    # Calculate per-entity metrics
    for entity_type, counts in base_metrics['entity_metrics'].items():
        counts['accuracy'] = counts['correct'] / counts['count'] if counts['count'] > 0 else 0
        entity_label = tag_to_index[entity_type]
        binary_true = [1 if label == entity_label else 0 for label in all_labels]
        binary_pred = [1 if label == entity_label else 0 for label in all_preds]
        counts['precision'] = precision_score(binary_true, binary_pred,
                                           average='binary', zero_division=0)
        counts['recall'] = recall_score(binary_true, binary_pred,
                                     average='binary', zero_division=0)
        counts['f1'] = f1_score(binary_true, binary_pred, average='binary', zero_division=0)
    if not bootstrap:
        return base_metrics

    # Bootstrap calculations
    n_samples = len(all_labels)
    bootstrap_metrics = {
        'accuracy': [],
        'f1_macro': [],
        'f1_micro': [],
        'entity_metrics': {entity: {
            'accuracy': [], 'precision': [], 'recall': [], 'f1': []
        } for entity in tag_to_index}
    }
    for _ in range(n_iterations):
        # Sample with replacement
        indices = np.random.randint(0, n_samples, size=n_samples)
        boot_true = [all_labels[i] for i in indices]
        boot_pred = [all_preds[i] for i in indices]
        # Calculate metrics for this bootstrap sample
        bootstrap_metrics['accuracy'].append(accuracy_score(boot_true, boot_pred))
        bootstrap_metrics['f1_macro'].append(f1_score(boot_true, boot_pred, average='macro'))
        bootstrap_metrics['f1_micro'].append(f1_score(boot_true, boot_pred, average='micro'))
        # Calculate entity-level metrics for this bootstrap sample
        for entity_type in tag_to_index:
            entity_label = tag_to_index[entity_type]
            binary_true = [1 if label == entity_label else 0 for label in boot_true]
            binary_pred = [1 if label == entity_label else 0 for label in boot_pred]
            entity_correct = sum(1 for t, p in zip(boot_true, boot_pred)
                               if t == entity_label and t == p)
            entity_total = sum(1 for t in boot_true if t == entity_label)
            bootstrap_metrics['entity_metrics'][entity_type]['accuracy'].append(
                entity_correct / entity_total if entity_total > 0 else 0)
            bootstrap_metrics['entity_metrics'][entity_type]['precision'].append(
                precision_score(binary_true, binary_pred, average='binary', zero_division=0))
            bootstrap_metrics['entity_metrics'][entity_type]['recall'].append(
                recall_score(binary_true, binary_pred, average='binary', zero_division=0))
            bootstrap_metrics['entity_metrics'][entity_type]['f1'].append(
                f1_score(binary_true, binary_pred, average='binary', zero_division=0))

    # Calculate confidence intervals and add to base_metrics
    for metric in ['accuracy', 'f1_macro', 'f1_micro']:
        base_metrics[f'{metric}_95CI_lower'] = np.percentile(bootstrap_metrics[metric], 2.5)
        base_metrics[f'{metric}_95CI_upper'] = np.percentile(bootstrap_metrics[metric], 97.5)
        base_metrics[f'{metric}_95CI_median'] = np.percentile(bootstrap_metrics[metric], 50)

    for entity_type in tag_to_index:
        for metric in ['accuracy', 'precision', 'recall', 'f1']:
            values = bootstrap_metrics['entity_metrics'][entity_type][metric]
            base_metrics['entity_metrics'][entity_type][f'{metric}_95CI_lower'] = \
                np.percentile(values, 2.5)
            base_metrics['entity_metrics'][entity_type][f'{metric}_95CI_upper'] = \
                np.percentile(values, 97.5)
            base_metrics['entity_metrics'][entity_type][f'{metric}_95CI_median'] = \
                np.percentile(values, 50)

    return base_metrics

def calculate_re_metrics(all_preds, all_labels, bootstrap=False, n_iterations=100):
    """
    Calculate RE metrics with optional bootstrap confidence intervals
    
    Args:
        all_preds: Predicted labels
        all_labels: True labels
        bootstrap: Whether to calculate bootstrap CIs
        n_iterations: Number of bootstrap iterations
    """
    base_metrics = {
        'accuracy': accuracy_score(all_labels, all_preds),
        'f1': f1_score(all_labels, all_preds),
        'precision': precision_score(all_labels, all_preds),
        'recall': recall_score(all_labels, all_preds),
        'auroc': roc_auc_score(all_labels, all_preds)
    }
    if not bootstrap:
        return base_metrics

    # Bootstrap calculations
    n_samples = len(all_labels)
    bootstrap_metrics = {
        'accuracy': [],
        'f1': [],
        'precision': [],
        'recall': [],
        'auroc': []
    }
    for _ in range(n_iterations):
        # Sample with replacement
        indices = np.random.randint(0, n_samples, size=n_samples)
        boot_true = [all_labels[i] for i in indices]
        boot_pred = [all_preds[i] for i in indices]

        # Calculate metrics for this bootstrap sample
        bootstrap_metrics['accuracy'].append(accuracy_score(boot_true, boot_pred))
        bootstrap_metrics['f1'].append(f1_score(boot_true, boot_pred))
        bootstrap_metrics['precision'].append(precision_score(boot_true, boot_pred))
        bootstrap_metrics['recall'].append(recall_score(boot_true, boot_pred))
        bootstrap_metrics['auroc'].append(roc_auc_score(boot_true, boot_pred))

    # Calculate confidence intervals and add to base_metrics
    for metric in ['accuracy', 'f1', 'precision', 'recall', 'auroc']:
        base_metrics[f'{metric}_95CI_lower'] = np.percentile(bootstrap_metrics[metric], 2.5)
        base_metrics[f'{metric}_95CI_upper'] = np.percentile(bootstrap_metrics[metric], 97.5)
        base_metrics[f'{metric}_95CI_median'] = np.percentile(bootstrap_metrics[metric], 50)

    return base_metrics

def load_best_model(save_dir, model_type, device):
    """Load the best model based on the saved metrics"""
    model_pattern = f'{model_type}_bs*_lr*_hd*_dropout*_wd*.pt'
    model_files = list(save_dir.glob(model_pattern))
    if not model_files:
        print("Found hyper_params.pkl but cannot find any model .pt file")
        return None, None, None
    best_model_file = max(model_files, key=os.path.getctime)
    checkpoint = torch.load(best_model_file, map_location=device)
    model = checkpoint['model']
    model.to(device)
    return model, checkpoint['hyperparameters'], checkpoint['metrics']

def inference(model, test_loader, device, model_type='BasicMLP'):
    """inference"""
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in test_loader:
            inputs, labels = batch[:2]
            inputs, labels = inputs.to(device), labels.to(device)
            if model_type == 'BiLSTM+CRF':
                mask = batch[2].to(device) if len(batch) > 2 else None
                preds = model(inputs, mask=mask)  # CRF decode returns list of lists
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().tolist())
            else:
                outputs = model(inputs)
                preds = outputs.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

    return all_preds, all_labels

def check_existing_hyperparameters(save_dir, hyperparameters):
    """Check if hyperparameter tuning has been done and load best models if so."""
    hyper_params_file = save_dir / 'hyper_params.pkl'
    if os.path.exists(hyper_params_file):
        with open(hyper_params_file, 'rb') as f:
            saved_hyper_params = pickle.load(f)
        if saved_hyper_params == hyperparameters:
            print("Hyperparameter tuning with the same conditions has already been performed.")
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            best_ner_model, ner_params, ner_metrics = load_best_model(save_dir, 'ner', device)
            best_re_model, re_params, re_metrics = load_best_model(save_dir, 're', device)
            if best_ner_model and best_re_model:
                print("\nLoaded Best NER Model:")
                print(f"Hyperparameters: {ner_params}")
                print_metrics(ner_metrics, {})
                print("\nLoaded Best RE Model:")
                print(f"Hyperparameters: {re_params}")
                print_metrics({}, re_metrics)
                return best_ner_model, best_re_model

    return False, False

def set_subgroups_from_to_and_size_n(subgroup_category):
    """set_subgroups_from_and_size_n"""
    if subgroup_category=='age':
        subgroups_from = ['under_65', 'over_65']
        subgroups_to = ['under_65', 'over_65']
        size_n = 700
    elif subgroup_category=='cancer_diagnosis':
        subgroups_from = ['cancer_blood_only', 'cancer_digestive_only']
        subgroups_to = ['cancer_blood_only', 'cancer_digestive_only',
                        'cancer_respiratory_only', 'cancer_breast_only',
                        'cancer_unknown_only', 'no_cancer']
        size_n = 200
    elif subgroup_category=='doctor_id_depart':
        subgroups_from = ['General Internal Medicine', 'Surgery']
        subgroups_to = ['General Internal Medicine', 'Surgery', 'Others']
        size_n = 150
    elif subgroup_category=='document_year':
        subgroups_from = [2010, 2015, 2020]
        subgroups_to = [2010, 2015, 2020]
        size_n = 400
    elif subgroup_category=='notetype':
        subgroups_from = ['S', 'O', 'A', 'P']
        subgroups_to = ['S', 'O', 'A', 'P']
        size_n = 250
    elif subgroup_category=='sex':
        subgroups_from = ['M', 'F']
        subgroups_to = ['M', 'F']
        size_n = 800
    elif subgroup_category=='textkorratio':
        subgroups_from = ['under_10%', 'over_10%']
        subgroups_to = ['under_10%', 'over_10%']
        size_n = 700
    elif subgroup_category=='text_len':
        subgroups_from = ['under_median_342', 'over_median_342']
        subgroups_to = ['under_median_342', 'over_median_342']
        size_n = 800
    else:
        print(f"Undefined subgroup_category ({subgroup_category})!")
        subgroups_from = subgroups_to = [None]
        size_n = None

    return subgroups_from, subgroups_to, size_n

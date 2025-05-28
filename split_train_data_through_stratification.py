"""Stratify train_data into train-valid-test through stratification"""
import contextlib
from functools import wraps
from typing import List, Tuple, Callable
import random

import pickle
from collections import Counter

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from settings import PPS



# Global variables
STRATIFY_COLUMNS = ['gender', 'note_type', 'document_year', 'age_group']
SPLIT_RATIOS = [0.7, 0.1, 0.2]
RANDOM_STATE = 42
random.seed(RANDOM_STATE)
SPLIT_RATIO_STR = "-".join([f"{ratio:.1f}".replace("0.", "") for ratio in SPLIT_RATIOS])
FNAME_TXT = f'{PPS.dir_train_valid_test_data}/stratified_split_log_{SPLIT_RATIO_STR}_without _with_doctor_ids.txt'


# Functions
def stratified_split(df: pd.DataFrame,
                     stratify_cols: List[str],
                     split_ratios: List[float],
                     random_state: int = 42) -> List[pd.DataFrame]:
    """
    Perform a stratified split of the dataframe into multiple sets.

    Args:
        df (pd.DataFrame): Input dataframe
        stratify_cols (List[str]): Columns to use for stratification
        split_ratios (List[float]): Ratios for each split (should sum to 1)
        random_state (int): Random state for reproducibility

    Returns:
        List[pd.DataFrame]: List of split dataframes
    """
    def _split_with_single_member_handling(df: pd.DataFrame,
                                           test_size: float,
                                           random_state: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
        array_strat_col = df['strat_col']
        class_counts = Counter(array_strat_col)
        single_member_classes = [c for c, count in class_counts.items() if count == 1]
        if single_member_classes:
            print(f"Warning: The following classes have only one member: {single_member_classes}")
            print("These will be included in the first split.")
            single_mask = array_strat_col.isin(single_member_classes)
            df_single = df[single_mask]
            df_multi = df[~single_mask]
            array_strat_col_new = array_strat_col[~single_mask]
            df_first, df_rest = train_test_split(df_multi, test_size=test_size,
                                                 stratify=array_strat_col_new,
                                                 random_state=random_state)
            df_first = pd.concat([df_first, df_single])
            return df_first, df_rest
        return train_test_split(df, test_size=test_size,
                                stratify=array_strat_col, random_state=random_state) 
    assert abs(sum(split_ratios) - 1) < 1e-6, "Split ratios must sum to 1 (or very close to 1)"

    df.loc[:, 'strat_col'] = df[stratify_cols].astype(str).agg('-'.join, axis=1)
    splits = []
    remaining_data = df.copy()
    for i, ratio in enumerate(split_ratios[:-1]):  # Process all but the last split
        if i == 0:
            split_size = ratio
        else:
            split_size = ratio / (1 - sum(split_ratios[:i]))
        split, remaining_data = _split_with_single_member_handling(
            remaining_data, test_size=(1 - split_size), random_state=random_state
        )
        splits.append(split)
    # Add the last split
    splits.append(remaining_data)
    # Remove the 'strat_col' from all splits
    for split in splits:
        split.drop('strat_col', axis=1, inplace=True)

    return splits

def calculate_statistics(data_split, split_name):
    """Calculate statistics for a given data split"""
    entities_count = [len(d['entities']) for d in data_split]
    relations_count = [len(d['relations']) for d in data_split]
    total_entities = sum(entities_count)
    total_relations = sum(relations_count)
    avg_entities = np.mean(entities_count)
    avg_relations = np.mean(relations_count)
    print(f"\n{split_name} Data Statistics:")
    print(f"Number of documents: {len(data_split)}")
    print(f"Total number of entities: {total_entities}")
    print(f"Total number of relations: {total_relations}")
    print(f"Average number of entities per document: {avg_entities:.2f}")
    print(f"Average number of relations per document: {avg_relations:.2f}")

def split_df_infos_through_stratification(df_infos, split_ratios):
    """Split train_data through stratification"""
    # Perform stratified split
    splits_df_infos = stratified_split(df_infos,
                                       STRATIFY_COLUMNS,
                                       split_ratios,
                                       random_state=RANDOM_STATE)

    # Print split sizes
    for i, split in enumerate(splits_df_infos):
        print(f"Split {i+1} size: {len(split)}")

    # Print distribution of stratify columns
    for col in STRATIFY_COLUMNS:
        print(f"\nDistribution of {col}:")
        for i, split in enumerate(splits_df_infos):
            print(f"Split {i+1}:", split[col].value_counts())

    return splits_df_infos

def log_to_file(log_file_path: str):
    """
    Decorator to redirect stdout to a file while executing the decorated function.
    
    Args:
        log_file_path (str): The path to the log file.
    
    Returns:
        A decorator function.
    """
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            @contextlib.contextmanager
            def redirect_stdout_to_file(file_path: str):
                with open(file_path, 'w', encoding='utf-8') as file:
                    with contextlib.redirect_stdout(file):
                        yield
            with redirect_stdout_to_file(log_file_path):
                result = func(*args, **kwargs)
            return result, log_file_path
        return wrapper
    return decorator

@log_to_file(FNAME_TXT)
def main():
    """Main function to process and split the data"""
    # Load data
    document_ids_inhouse = [data_i['document_id'] for data_i in PPS.data_inhouse]
    document_ids_void_entities_inhouse = \
        [data_i for data_i in PPS.data_inhouse if not data_i['entities']]
    assert len(document_ids_inhouse) == len(set(document_ids_inhouse)), \
        "Still found duplicate document_ids!"
    assert len(document_ids_void_entities_inhouse) == 0, "Found data of void entities!"

    # Load stratified info
    df_inhouse_infos_filtered = \
        PPS.df_inhouse_infos[PPS.df_inhouse_infos['document_id'].isin(document_ids_inhouse)]
    print(f"Total number of documents (by document id): {len(df_inhouse_infos_filtered)}")

    # Import document_ids with doctor_ids 
    with open(r"E:\NRF_CCADD\DATASET\241025\documentid2doctorid.pkl", 'rb') as f:
        documentid2doctorid = pickle.load(f)
    document_ids_with_doctor_id = {int(document_id.split('_')[1])
                                    for document_id, doctor_id in documentid2doctorid.items()
                                    if doctor_id != 'not_found'}

    # Split df_inhouse_infos_filtered by whether ther are with doctor_ids
    total_sample_n = len(df_inhouse_infos_filtered)
    df_inhouse_infos_without_doctor_ids = \
        df_inhouse_infos_filtered[[document_id not in document_ids_with_doctor_id
                                    for document_id in
                                        df_inhouse_infos_filtered['document_id']]]
    df_inhouse_infos_with_doctor_ids = \
        df_inhouse_infos_filtered[[document_id in document_ids_with_doctor_id
                                    for document_id in
                                        df_inhouse_infos_filtered['document_id']]]
    assert len(SPLIT_RATIOS) == 3, "The length of SPLIT_RATIOS must be equal to 3!"
    split_ratios_for_with_doctor_ids = \
        (len(df_inhouse_infos_with_doctor_ids)
            - int(total_sample_n*SPLIT_RATIOS[1]) - int(total_sample_n*SPLIT_RATIOS[2]),
         int(total_sample_n*SPLIT_RATIOS[1]),
         int(total_sample_n*SPLIT_RATIOS[2]))
    total = sum(split_ratios_for_with_doctor_ids)
    split_ratios_for_with_doctor_ids = tuple(n/total for n in split_ratios_for_with_doctor_ids)
        
    assert total + len(df_inhouse_infos_without_doctor_ids) == \
        total_sample_n, "Check split_ratios_for_with_doctor_ids!"

    # Set splits_df_infos (split through stratification)
    splits_df_infos = split_df_infos_through_stratification(df_inhouse_infos_with_doctor_ids,
                                                            split_ratios_for_with_doctor_ids)

    # Set splits_data
    splits_data = []
    for idx, split_df_infos in enumerate(splits_df_infos):
        if idx == 0: #training dataset
            document_ids = set(split_df_infos['document_id']).union(set(df_inhouse_infos_without_doctor_ids['document_id']))
        else:
            document_ids = set(split_df_infos['document_id'])
        split_data = [d for d in PPS.data_inhouse if d['document_id'] in document_ids]
        for data_i in split_data:
            data_i.update({'doctor_id':
                documentid2doctorid[f"document_{data_i['document_id']}"]})
        random.shuffle(split_data)
        splits_data.append(split_data)

    # Calculate and print statistics for each data split
    for i, split in enumerate(splits_data):
        calculate_statistics(split, f"Split {i+1}")
    calculate_statistics([d for split_data in splits_data for d in split_data], "Total data")

    # Save splits_data and train_valid_test_data_in_a_single_list
    ratio_str = SPLIT_RATIO_STR
    p_fname = f'{PPS.dir_train_valid_test_data}/splits_data_{ratio_str}_without _with_doctor_ids.pkl'
    with open(p_fname, 'wb') as file:
        pickle.dump(splits_data, file)

    assert all(set(len(d['annotators']) for d in split) == {1} for split in splits_data), \
        "For all data, annotators must be one!"

if __name__ == "__main__":
    _, log_file_path = main()
    print(f"Log file has been saved as '{log_file_path}'")
    # Print the contents of the log file
    with open(log_file_path, 'r', encoding='utf-8') as f:
        print(f.read())

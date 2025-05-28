import pickle
import random
import secrets
import os
from itertools import product
from typing import List, Dict, Tuple
from datetime import datetime
import hashlib
import json

from collections import defaultdict

from langchain.embeddings import HuggingFaceEmbeddings
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi
import faiss

import numpy as np

from utils import (
    DOC_INFOS,
    set_entity_labels_for_entity_group_sentence,
    set_task_description,
    set_confusing_cases,
    create_tagged_text_with_priority,
    format_output_ner,
    set_patient_summaries_for_test_data_i,
)
from settings import PPS

import sys
import torch

# Global variables
DIR_BASE = r'C:\Users\username\Documents\projects\ICLforclinicalIE'
RANDOM_SEED = 42
RANDOM_RANDOM = True

TEST_DATA_N = 200

#Main class - NERPromptGenerator
class NERPromptGenerator:
    """NERPromptGenerator"""
    def __init__(self,
                 test_data:List[Dict],
                 train_data: List[Dict],
                 embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 use_gpu: bool = False,
                 subgroup_category: str = None,
                 subgroup_from: str = None,
                 subgroup_to: str = None,
                 train_n_small_int: int = None,
                 using_masked_text: bool = False):
        self.test_data = test_data
        self.train_data = train_data
        self.embedding_model = embedding_model
        self.use_gpu = use_gpu
        self.device = "cuda" if use_gpu else "cpu"
        
        # Use default HuggingFace embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': self.device}
        )

        self.using_masked_text = using_masked_text
        self.demonstration_sorting_methods = {
            "descending": lambda x: x,
            "ascending": lambda x: list(reversed(x)),
            "bookend": self._sort_bookend,
            "reverse_bookend": self._sort_reverse_bookend,
            "alternating": self._sort_alternating,
            "wave": self._sort_wave
        }

        #re-select train-test data by subgroup
        self.subgroup_category = subgroup_category
        self.subgroup_from = subgroup_from
        self.subgroup_to = subgroup_to
        self.train_n_small_int = train_n_small_int
        self.test_texts = [example['text'] for example in test_data]
        self.test_texts_index = [example['document_id'] for example in test_data]
        self.train_texts = [example['text'] for example in train_data]
        self.train_texts_index = [example['document_id'] for example in train_data]
        #set train-test data/text for masked texts
        if using_masked_text:
            self.masked_text = PPS.masked_texts
            self.test_texts_masked = [PPS.masked_texts[index]
                                        for index in self.test_texts_index]
            self.train_texts_masked = [PPS.masked_texts[index]
                                        for index in self.train_texts_index]
            self.test_data_masked = []
            for td, ttm in zip(test_data, self.test_texts_masked):
                td_masked = td.copy()
                td_masked['text'] = ttm
                self.test_data_masked.append(td_masked)
            self.train_data_masked = []
            for td, ttm in zip(train_data, self.train_texts_masked):
                td_masked = td.copy()
                td_masked['text'] = ttm
                self.train_data_masked.append(td_masked)
            assert len(self.test_texts_masked)==len(self.test_data), \
                "Length of masked test data must be same with test data!"
            assert len(self.train_texts_masked)==len(self.train_data), \
                "Length of masked test data must be same with train data!"
        else:
            self.masked_text =  [''] * len(PPS.masked_texts)

        if 'Embedding' in PPS.demonstration_selection_methods:
            print("Precompute embeddings for train_data_total!")
            if using_masked_text:
                self.train_embeddings = self.embeddings.embed_documents(self.train_texts_masked)
            else:
                self.train_embeddings = self.embeddings.embed_documents(self.train_texts)
            self.embedding_size = len(self.train_embeddings[0])
            print("Finish getting embeddings for train_data_total!")
            self.index = faiss.IndexFlatIP(self.embedding_size)
            if self.use_gpu:
                try:
                    res = faiss.StandardGpuResources()
                    self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
                    print("Using GPU for FAISS")
                except AttributeError:
                    print("GPU support not available for FAISS. Falling back to CPU.")
                    self.use_gpu = False
            self.index.add(np.array(self.train_embeddings))

        if 'TF-IDF' in PPS.demonstration_selection_methods:
            print("Precompute TF-IDF for train_data_total!")
            self.tfidf_vectorizer = TfidfVectorizer()
            if using_masked_text:
                self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.train_texts_masked)
            else:
                self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.train_texts)
            print("Finish getting TF-IDF for train_data_total!")

        if 'BM25' in PPS.demonstration_selection_methods:
            print("Precompute bm25 for train_data_total!")
            if using_masked_text:
                tokenized_corpus = [text.split() for text in self.train_texts_masked]
            else:
                tokenized_corpus = [text.split() for text in self.train_texts]
            self.bm25 = BM25Okapi(tokenized_corpus)
            print("Finish getting bm25 for train_data_total!")

    def generate_ner_prompts(
        self,
        entity_group: bool,
        sentence_entity: bool,
        add_patient_infos: bool,
        patient_infos_categories: List[str],
        output_type_ner: List[str],
        demonstration_selelction_method: List[str],
        demonstration_sample_n: List[int],
        demonstration_sorting_method: List[str],
        sample_n_from: int,
        using_text_with_linebreaks: bool
    ) -> List[Tuple[str, int, Dict]]:
        """generate_ner_prompts"""
        final_input_prompts = []
        if len(self.test_data)==0:
            return None
        for test_data_i in self.test_data:
            # generate_final_input_prompt_ner
            final_input_prompt = \
                self.generate_final_input_prompt_ner(output_type_ner=output_type_ner,
                                                     test_data_i=test_data_i,
                                                     entity_group=entity_group,
                                                     sentence_entity=sentence_entity,
                                                     add_patient_infos=add_patient_infos,
                                                     patient_infos_categories=\
                                                         patient_infos_categories,
                                                     demonstration_selelction_method=\
                                                         demonstration_selelction_method,
                                                     demonstration_sorting_method=\
                                                         demonstration_sorting_method,
                                                     demonstration_sample_n=\
                                                         demonstration_sample_n,
                                                     sample_n_from=sample_n_from,
                                                     using_text_with_linebreaks=\
                                                         using_text_with_linebreaks)
            final_input_prompts.append(final_input_prompt)
        return final_input_prompts

    def select_train_data_for_demonstrations(self,
                                             test_data_i: Dict,
                                             method: str,
                                             count: int,
                                             sorting_method: str = "descending",
                                             sample_n_from: int = None) -> List[Dict]:
        """select_train_data_for_demonstrations"""
        if count <= 0:
            return []

        if method == "Random":
            if RANDOM_RANDOM:
                seed = secrets.randbits(32)
                random.seed(seed)
            else:
                random.seed(RANDOM_SEED)
            if sample_n_from:
                return random.sample(self.train_data[:sample_n_from],
                                     min(count, len(self.train_data[:sample_n_from])))
            return random.sample(self.train_data, min(count, len(self.train_data)))
        if method == "Embedding":
            examples = self.embedding_based_selection(test_data_i, count, sample_n_from)
        elif method == "TF-IDF":
            examples = self.tfidf_based_selection(test_data_i, count, sample_n_from)
        elif method == "BM25":
            examples = self.bm25_based_selection(test_data_i, count, sample_n_from)
        elif method == "Domain":
            examples = self.domain_based_selection(test_data_i, count, sample_n_from)
        else:
            raise ValueError(f"Unsupported few-shot selection method: {method}")

        return self.demonstration_sorting_methods[sorting_method](examples)

    def set_final_annotation_instruction_str(self,
                                             output_type_ner):
        """set_final_annotation_instruction_str"""
        final_annotation_instruction_str = ""
        current_i = 0
        for final_annotation_instruction in PPS.final_annotation_instructions:
            current_i += 1
            final_annotation_instruction_str += f"{current_i}. {final_annotation_instruction}"
        if output_type_ner=='entity_list_appearance_order_json_format':
            for final_annotation_instruction in \
                PPS.final_annotation_instruction_entity_list_json:
                current_i += 1
                final_annotation_instruction_str += \
                    f"{current_i}. {final_annotation_instruction}"
        if 'tagged_text' in output_type_ner:
            for final_annotation_instruction in PPS.final_annotation_instruction_tagged_text:
                current_i += 1
                final_annotation_instruction_str += \
                    f"{current_i}. {final_annotation_instruction}"

        return final_annotation_instruction_str

    def generate_final_input_prompt_ner(
        self,
        output_type_ner: str,
        test_data_i: Dict,
        entity_group: bool,
        sentence_entity: bool,
        add_patient_infos: bool,
        patient_infos_categories: List[str],
        demonstration_selelction_method: str,
        demonstration_sorting_method: str,
        demonstration_sample_n: int,
        sample_n_from: int,
        using_text_with_linebreaks: bool
    ) -> Tuple[str, int, Dict]:
        """generate_ner_prompt"""
        final_input_prompt = PPS.common_prompt_ner
        #set entity_labels
        entity_labels = set_entity_labels_for_entity_group_sentence(entity_group,
                                                                    sentence_entity)
        # Replace common_prompt_elements
        for element_type, element_string in PPS.common_prompt_elements.items():
            if element_type=='text_type_name':
                text_type_name = 'clinical notes' if 'inhouse' in PPS.document_source \
                    else 'discharge summary'
                final_input_prompt = final_input_prompt.replace(element_string,
                                                                text_type_name)

            elif element_type=='output_format_name':
                if output_type_ner=='entity_list_with_start_end':
                    output_format_name = 'entity list with start and end positions'
                elif output_type_ner=='entity_list_appearance_order_with_entity_label':
                    output_format_name = 'entity list with its own entity label'
                elif output_type_ner=='entity_list_appearance_order_json_format':
                    output_format_name = 'entity list in a list-dictionary format'
                elif output_type_ner=='tagged_text_with_abbreviations':
                    output_format_name = 'tagged text with abbreviations'
                elif output_type_ner=='tagged_text_without_abbrevaitions':
                    output_format_name = 'tagged text without abbreviations'
                final_input_prompt = final_input_prompt.replace(element_string,
                                                                output_format_name)

            elif element_type=='task_description':
                task_description = set_task_description(entity_labels=entity_labels,
                                                        entity_group=entity_group,
                                                        sentence_entity=sentence_entity)
                if 'with_abbreviations' in output_type_ner:
                    for entity_label in entity_labels:
                        abbr = PPS.abbreviations[entity_label]
                        task_description = \
                            task_description.replace(f"{entity_label}: ",
                                                     f"{entity_label} ({abbr}): ")
                final_input_prompt = final_input_prompt.replace(element_string,
                                                                task_description)

            elif element_type=='confusing_cases':
                confusing_cases = set_confusing_cases()
                if confusing_cases is not None:
                    final_input_prompt = final_input_prompt.replace(element_string,
                                                                    confusing_cases)

            elif element_type=='output_example':
                text = \
                    'blah blah entity-text-1 is blah blah. entity-text-2 is blah blah'
                sample_entities = \
                    [{'text': 'entity-text-1', 'label': 'entity-label-1',
                      'start': 10, 'end': 23},
                     {'text': 'entity-text-2', 'label': 'entity-label-2',
                      'start': 38, 'end': 51}]
                output_example = \
                    format_output_ner(output_type_ner=output_type_ner,
                                      entity_labels=[e['label'] for e in sample_entities],
                                      entities=sample_entities,
                                      text=text,
                                      only_for_formatting=True)
                final_input_prompt = final_input_prompt.replace(element_string,
                                                                output_example)

            elif element_type=='final_annotation_instructions':
                final_annotation_instruction_str = \
                    self.set_final_annotation_instruction_str(output_type_ner)
                final_input_prompt = \
                    final_input_prompt.replace(element_string,
                                               final_annotation_instruction_str)

            elif element_type=='selected_demonstration':
                selected_data_for_demonstrations = \
                    self.select_train_data_for_demonstrations(test_data_i=test_data_i,
                                                              method=\
                                                                  demonstration_selelction_method,
                                                              count=demonstration_sample_n,
                                                              sorting_method=\
                                                                  demonstration_sorting_method,
                                                              sample_n_from=sample_n_from)
                #set output_format_ner
                output_format_ner = PPS.output_formats_ner[output_type_ner]
                #set selected_demonstrations_str
                selected_demos = ""
                for i, data_for_demo in enumerate(selected_data_for_demonstrations):
                    text = data_for_demo['text']
                    entities = data_for_demo['entities']
                    # For entity grouping
                    if entity_group:
                        for entity in entities:
                            for entity_group_name, entities_group in PPS.entity_groups.items():
                                if entity['label'] in entities_group:
                                    entity['label'] = entity_group_name
                    # Filtering entities before foramtting
                    entities = [entity for entity in entities
                                    if entity['label'] in entity_labels]
                    #set formatted_output_ner
                    formatted_output_ner = format_output_ner(output_type_ner,
                                                             entity_labels,
                                                             entities,
                                                             text)
                    if 'entity_list' in output_type_ner:
                        selected_demo = output_format_ner.format(
                            document_type_str=text_type_name,
                            input_text=text,
                            entity_list=formatted_output_ner
                        )
                        selected_demo = \
                            selected_demo.replace(f'{text_type_name}>',
                                                  f'{text_type_name}> - Example {i + 1}')
                        selected_demo = \
                            selected_demo.replace('entity list>',
                                                  f'entity list> - Example (answer) {i + 1}')
                    elif 'tagged_text' in output_type_ner:
                        selected_demo = output_format_ner.format(
                            document_type_str=text_type_name,
                            input_text=text,
                            tagged_text=formatted_output_ner
                        )
                        selected_demo = \
                            selected_demo.replace(f'{text_type_name}>',
                                                  f'{text_type_name}> - Example {i + 1}')
                        selected_demo = \
                            selected_demo.replace('Tagged text>',
                                                  f'Tagged text> - Example (answer) {i + 1}')
                    else:
                        assert False, f"Undefined output_type_ner ({output_type_ner})!"
                    selected_demo += "\n"
                    selected_demos += selected_demo
                final_input_prompt = final_input_prompt.replace(element_string,
                                                                selected_demos)
            elif element_type=='patient_information_summaries':
                if add_patient_infos:
                    patient_summaries_str = \
                        set_patient_summaries_for_test_data_i(test_data_i,
                                                              PPS.df_patient_infos_summaries,
                                                              patient_infos_categories)
                    final_input_prompt = final_input_prompt.replace(element_string,
                                                                    patient_summaries_str)   
                else:
                    final_input_prompt = final_input_prompt.replace(element_string,
                                                                    "")                    
            elif element_type=='test_input_text':
                #set input_text - by using_text_with_linebreaks
                if using_text_with_linebreaks:
                    input_text = test_data_i['text_raw']
                else:
                    input_text = test_data_i['text']
                #set test_input_text
                if 'entity_list' in output_type_ner:
                    test_input_text = output_format_ner.format(
                        document_type_str=text_type_name,
                        input_text=input_text,
                        entity_list=[defaultdict(str)]
                    )
                    test_input_text = \
                        test_input_text.split('entity list>')[0] + 'entity list>\n'
                elif 'tagged_text' in output_type_ner:
                    test_input_text = output_format_ner.format(
                        document_type_str=text_type_name,
                        input_text=input_text,
                        tagged_text=[defaultdict(str)]
                    )
                    test_input_text = \
                        test_input_text.split('Tagged text>')[0] + 'Tagged text>\n'
                else:
                    assert False, f"Undefined output_type_ner ({output_type_ner})!"
                final_input_prompt = final_input_prompt.replace(element_string,
                                                                test_input_text)

        return final_input_prompt

    def embedding_based_selection(self,
                                  test_data: Dict,
                                  count: int,
                                  sample_n_from: int = None) -> List[Dict]:
        """
        embedding_based_selection with sample_n_from filtering
        
        Args:
            test_data: Dictionary containing test data
            count: Number of items to select
            sample_n_from: If not None, only select from first N training samples
            
        Returns:
            List of selected training data items
        """
        if count <= 0:
            return []  # Return an empty list if count is 0 or negative

        if self.using_masked_text:
            test_index = test_data['document_id']
            text_masked_i = PPS.masked_texts[test_index]
            test_embedding = self.embeddings.embed_query(text_masked_i)
        else:
            test_embedding = self.embeddings.embed_query(test_data['text'])

        # Get nearest neighbors with a larger K
        search_k = len(self.train_data)  # Get all possible neighbors first
        _, nearest_indices = self.index.search(np.array([test_embedding]), search_k)
        indices_list = nearest_indices[0]

        # Filter indices by sample_n_from if specified
        if sample_n_from is not None:
            indices_list = [idx for idx in indices_list if idx < sample_n_from]

        # Take only count number of items
        return [self.train_data[i] for i in indices_list[:count]]

    def tfidf_based_selection(self,
                              test_data: Dict,
                              count: int,
                              sample_n_from: int = None) -> List[Dict]:
        """
        tfidf_based_selection with sample_n_from filtering
        
        Args:
            test_data: Dictionary containing test data
            count: Number of items to select
            sample_n_from: If not None, only select from first N training samples
            
        Returns:
            List of selected training data items
        """
        if self.using_masked_text:
            test_index = test_data['document_id']
            text_masked_i = PPS.masked_texts[test_index]
            test_vector = self.tfidf_vectorizer.transform([text_masked_i])
        else:
            test_vector = self.tfidf_vectorizer.transform([test_data['text']])
        # Calculate similarities with all training samples
        similarities = (self.tfidf_matrix * test_vector.T).toarray().flatten()

        # Get all indices sorted by similarity
        sorted_indices = similarities.argsort()[::-1]
        # Filter indices by sample_n_from if specified
        if sample_n_from is not None:
            sorted_indices = [idx for idx in sorted_indices if idx < sample_n_from]
        # Take only count number of items
        top_indices = sorted_indices[:count]
        return [self.train_data[i] for i in top_indices]

    def bm25_based_selection(self,
                             test_data: Dict,
                             count: int,
                             sample_n_from: int = None) -> List[Dict]:
        """
        bm25_based_selection with sample_n_from filtering
        
        Args:
            test_data: Dictionary containing test data
            count: Number of items to select
            sample_n_from: If not None, only select from first N training samples
            
        Returns:
            List of selected training data items
        """
        if self.using_masked_text:
            test_index = test_data['document_id']
            text_masked_i = PPS.masked_texts[test_index]
            tokenized_query = text_masked_i.split()
        else:
            tokenized_query = test_data['text'].split()
        # Get scores for all training samples
        scores = self.bm25.get_scores(tokenized_query)

        # Get all indices sorted by score
        sorted_indices = scores.argsort()[::-1]
        # Filter indices by sample_n_from if specified
        if sample_n_from is not None:
            sorted_indices = [idx for idx in sorted_indices if idx < sample_n_from]
        # Take only count number of items
        top_indices = sorted_indices[:count]
        return [self.train_data[i] for i in top_indices]

    def domain_based_selection(self,
                            test_data: Dict,
                            count: int,
                            sample_n_from: int = None) -> List[Dict]:
        """
        Domain-based selection using pre-calculated domain rankings
        
        Args:
            test_data: Dictionary containing test data
            count: Number of items to select
            sample_n_from: If not None, only select from first N training samples
            
        Returns:
            List of selected training data items
        """
        if count <= 0:
            return []  # Return an empty list if count is 0 or negative
        # Path to the domain rankings file
        dir_domain_rankings_pickle = F"{DIR_BASE}\\save_results\\domain_rankings.pkl"

        # Check if domain rankings file exists
        if not os.path.exists(dir_domain_rankings_pickle):
            print(f"Domain rankings file not found at {dir_domain_rankings_pickle}")
            return []

        # Load domain rankings
        try:
            with open(dir_domain_rankings_pickle, 'rb') as f:
                domain_rankings = pickle.load(f)
        except Exception as e:
            print(f"Error loading domain rankings: {e}")
            return []
        # Get document_id from test_data
        test_doc_id = test_data.get('document_id')
        if test_doc_id is None:
            print("Test data does not have a document_id")
            return []
        # Get domain rankings for this test document
        if test_doc_id not in domain_rankings:
            print(f"No domain rankings found for document_id {test_doc_id}")
            return []

        # Get ranked document ids
        ranked_doc_ids = domain_rankings[test_doc_id]
        # Create a mapping from document_id to train_data index for quick lookup
        doc_id_to_index = {item.get('document_id'): idx for idx, item in enumerate(self.train_data)}
        # Filter out document IDs that don't exist in our train_data
        valid_doc_ids = [doc_id for doc_id in ranked_doc_ids if doc_id in doc_id_to_index]
        # Apply sample_n_from filtering if specified
        if sample_n_from is not None:
            # Filter to include only documents with indices less than sample_n_from
            valid_doc_ids = [doc_id for doc_id in valid_doc_ids
                                if doc_id_to_index[doc_id] < sample_n_from]
        
        # Take only the top 'count' items
        selected_doc_ids = valid_doc_ids[:count]
        
        # Convert document IDs to train_data items
        selected_train_data = [self.train_data[doc_id_to_index[doc_id]]
                                for doc_id in selected_doc_ids]
        
        return selected_train_data

    def _sort_bookend(self, examples):
        if len(examples) <= 2:
            return examples
        sorted_examples = []
        half = len(examples) // 2
        for i in range(half):
            sorted_examples.append(examples[i])
            if i < len(examples) - 1 - i:
                sorted_examples.append(examples[len(examples) - 1 - i])
        if len(examples) % 2 == 1:
            sorted_examples.append(examples[half])
        return sorted_examples

    def _sort_reverse_bookend(self, examples):
        if len(examples) <= 2:
            return examples
        examples = list(reversed(examples))
        return self._sort_bookend(examples)

    def _sort_alternating(self, examples):
        if len(examples) <= 1:
            return examples
        sorted_examples = []
        half = (len(examples) + 1) // 2
        for i in range(half):
            sorted_examples.append(examples[i])
            if i + half < len(examples):
                sorted_examples.append(examples[i + half])
        return sorted_examples

    def _sort_wave(self, examples):
        if len(examples) <= 2:
            return examples
        sorted_examples = []
        third = len(examples) // 3
        for i in range(third):
            if i < len(examples) - 2*i:
                sorted_examples.append(examples[i])
            if i + third < len(examples):
                sorted_examples.append(examples[i + third])
            if i + 2*third < len(examples):
                sorted_examples.append(examples[len(examples) - 1 - i])
        return sorted_examples

    def create_metadata(self,
                        entity_group,
                        sentence_entity,
                        validation_data,
                        add_patient_infos,
                        patient_infos_categories,
                        output_type_ner,
                        demonstration_selelction_method,
                        demonstration_sample_n,
                        demonstration_sorting_method,
                        sample_n_from,
                        lambda_,
                        alpha,
                        ann_pool_method):
        """Create metadata and generate simple filename"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        # Store all conditions in metadata
        metadata = {
            'timestamp': timestamp,
            'conditions': {
                'output_type_ner': output_type_ner,
                'demonstration_method': demonstration_selelction_method,
                'sample_n': demonstration_sample_n,
                'sorting_method': demonstration_sorting_method,
                'test_data_n': len(self.test_data),
                'validation_data': validation_data,
                'add_patient_infos': add_patient_infos,
                'patient_infos_categories': patient_infos_categories,
                'using_masked_text': self.using_masked_text,
                'sentence_entity': sentence_entity,
                'entity_group': entity_group,
                'subgroup_category': self.subgroup_category,
                'subgroup_from': self.subgroup_from, 
                'subgroup_to': self.subgroup_to,
                'sample_n_from': sample_n_from
            }
        }
        if lambda_ is not None:
            metadata['conditions']['ann_pool_lambda'] = lambda_
        if alpha is not None:
            metadata['conditions']['ann_pool_alpha'] = alpha
        if ann_pool_method is not None:
            metadata['conditions']['ann_pool_method'] = ann_pool_method

        # Generate short filename using hash
        file_hash = hashlib.md5(str(metadata['conditions']).encode()).hexdigest()[:8]
        simple_filename = f"ner_{timestamp}_{file_hash}.pkl"
        metadata['filename'] = simple_filename

        return metadata, simple_filename

    def set_pkl_filename_for_final_input_prompt_ner(self,
                                                    entity_group,
                                                    sentence_entity,
                                                    validation_data,
                                                    add_patient_infos,
                                                    patient_infos_categories,
                                                    output_type_ner,
                                                    demonstration_selelction_method,
                                                    demonstration_sample_n,
                                                    sample_n_from=None,
                                                    demonstration_sorting_method="descending",
                                                    lambda_=None,
                                                    alpha=None,
                                                    ann_pool_method=None):
        """set_filename_for_final_input_prompt"""
        test_data_n = len(self.test_data)
        if demonstration_sample_n == 0:
            pkl_filename = \
                f"{output_type_ner}_none_examples0_samplen{test_data_n}"
        elif demonstration_selelction_method == "Random":
            pkl_filename = \
                f"{output_type_ner}_{demonstration_selelction_method}_examples{demonstration_sample_n}_samplen{test_data_n}"
        else:
            pkl_filename = \
                f"{output_type_ner}_{demonstration_selelction_method}_sorting{demonstration_sorting_method}_examples{demonstration_sample_n}_samplen{test_data_n}"
        if validation_data:
            pkl_filename += '_validation'
        if add_patient_infos:
            pkl_filename += '_addptntinfos'
            pkl_filename += f'{len(patient_infos_categories)}'
        if self.using_masked_text:
            pkl_filename += '_maskedtext'
        if sentence_entity:
            pkl_filename += '_sentenceentity'
        if entity_group:
            pkl_filename += '_entitygroup'
        if self.subgroup_category:
            pkl_filename += f'_subgroupcate{self.subgroup_category}'
        if self.subgroup_from:
            pkl_filename += f'_subgroupfrom{self.subgroup_from}'
        if self.subgroup_to:
            pkl_filename += f'_subgroupto{self.subgroup_to}'
        if sample_n_from:
            pkl_filename += f'_samplenfrom{sample_n_from}'
        if lambda_:
            pkl_filename += f'_selectiveannpool{lambda_}'
            if ann_pool_method!='BM25':
                raise ValueError("Selective annotation pooling is only supported with BM25 method.")
        if alpha:
            pkl_filename += f'_alpha{alpha}'
            if ann_pool_method!='BM25':
                raise ValueError("Selective annotation pooling is only supported with BM25 method.")

        pkl_filename += '.pkl'

        return pkl_filename

    def generate_and_save_all_combinations(self,
                                           output_types_ner,
                                           demonstration_selection_methods,
                                           demonstration_sample_ns,
                                           demonstration_sorting_methods,
                                           date_ver,
                                           samples_n_from=[None],
                                           entity_group=False,
                                           sentence_entity=False,
                                           validation_data=False,
                                           add_patient_infos=False,
                                           patient_infos_categories=[],
                                           using_text_with_linebreaks=False,
                                           document_source='inhouse',
                                           lambda_=None,
                                           alpha=None,
                                           ann_pool_method=None):
        """generate_and_save_all_combinations"""
        # Set save_dir
        save_dir = os.path.join(PPS.dir_final_input_prompts, date_ver)
        if using_text_with_linebreaks:
            save_dir += '_textwithlinbreaks'
        if document_source=='discharge_summary':
            save_dir += '_ds'
        os.makedirs(save_dir, exist_ok=True)

        if self.train_n_small_int:
            self.train_data = random.sample(self.train_data,
                                            min(self.train_n_small_int, len(self.train_data)))
            samples_n_from = [self.train_n_small_int]
        print(f"   - samples_n_from len, final: {samples_n_from}")

        for (output_type_ner,
             demonstration_selelction_method,
             demonstration_sample_n,
             sample_n_from) in product(output_types_ner,
                                       demonstration_selection_methods,
                                       demonstration_sample_ns,
                                       samples_n_from):
            if demonstration_sample_n == 0 or demonstration_selelction_method == "Random":
                sorting_methods = ["descending"]
            else:
                sorting_methods = demonstration_sorting_methods
            for sorting_method in sorting_methods:
                #check sample_n_from is valid
                if sample_n_from is None or len(self.train_data)>=sample_n_from:
                    pkl_filename = self.set_pkl_filename_for_final_input_prompt_ner(
                        entity_group=entity_group,
                        sentence_entity=sentence_entity,
                        validation_data=validation_data,
                        add_patient_infos=add_patient_infos,
                        patient_infos_categories=patient_infos_categories,
                        output_type_ner=output_type_ner,
                        demonstration_selelction_method=demonstration_selelction_method,
                        demonstration_sample_n=demonstration_sample_n,
                        demonstration_sorting_method=sorting_method,
                        sample_n_from=sample_n_from,
                        lambda_=lambda_,
                        alpha=alpha,
                        ann_pool_method=ann_pool_method
                    )
                    filepath = os.path.join(save_dir, pkl_filename)
                    if os.path.exists(filepath):
                        print(f"File already exists: {filepath}")
                        continue

                    final_input_prompts = self.generate_ner_prompts(
                        entity_group=entity_group,
                        output_type_ner=output_type_ner,
                        sentence_entity=sentence_entity,
                        add_patient_infos=add_patient_infos,
                        patient_infos_categories=patient_infos_categories,
                        demonstration_selelction_method=demonstration_selelction_method,
                        demonstration_sample_n=demonstration_sample_n,
                        demonstration_sorting_method=sorting_method,
                        sample_n_from=sample_n_from,
                        using_text_with_linebreaks=using_text_with_linebreaks
                    )

                    # Add metadata file path after save_dir setup
                    metadata_path = os.path.join(save_dir, 'metadata.json')
                    # Inside the file saving block
                    metadata, simple_filename = self.create_metadata(
                        entity_group=entity_group,
                        sentence_entity=sentence_entity,
                        validation_data=validation_data,
                        add_patient_infos=add_patient_infos,
                        patient_infos_categories=patient_infos_categories,
                        output_type_ner=output_type_ner,
                        demonstration_selelction_method=demonstration_selelction_method,
                        demonstration_sample_n=demonstration_sample_n,
                        demonstration_sorting_method=sorting_method,
                        sample_n_from=sample_n_from,
                        lambda_=lambda_,
                        alpha=alpha,
                        ann_pool_method=ann_pool_method,
                    )
                    filepath = os.path.join(save_dir, simple_filename)

                    # Update metadata file
                    try:
                        with open(metadata_path, 'r') as f:
                            all_metadata = json.load(f)
                    except FileNotFoundError:
                        all_metadata = []
                    # Check if metadata with same conditions already exists
                    duplicate_exists = any(
                        existing['conditions'] == metadata['conditions']
                        for existing in all_metadata
                    )

                    # Only append if no duplicate exists
                    if not duplicate_exists:
                        all_metadata.append(metadata)
                        with open(metadata_path, 'w') as f:
                            json.dump(all_metadata, f, indent=2)

                    # Save the actual data
                    final_input_prompts_and_test_entities = \
                        list(zip(final_input_prompts, self.test_data))
                    with open(filepath, 'wb') as f:
                        pickle.dump(final_input_prompts_and_test_entities, f)
                    print(f"Saved results to {filepath}")

                else:
                    print(f"sample_n_from ({sample_n_from}) is larger than train_data size ({len(self.train_data)}!)")

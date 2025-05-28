"""Set final prompts for LLM testing"""
import random
from datetime import datetime

from prompt_generator import NERPromptGenerator
from settings import PPS

from utils import (
    DOC_INFOS,
    convert_negation_labels_for_check,
    import_train_valid_test_data_by_source,
    import_df_annotated_data
    )
random.seed(42)
TEST_SAMPLE_N = 200

def set_final_prompts(document_source='inhouse',
                      test_sampel_size=None,
                      date_ver=datetime.now().strftime('%y%m%d'),
                      entity_group=False,
                      sentence_entity=False,
                      using_masked_text=False,
                      subgroup_category=None,
                      subgroup_from_i=None,
                      subgroup_to_i=None,
                      samples_n_from=None,
                      add_patient_infos=False,
                      patient_infos_categories=[
                          'summary_basic_with_note',
                          'summary_diagnosis',
                          'summary_long_term_prescription',
                          'summary_short_term_prescription',
                          'summary_recent_prescription',
                          'summary_lab_results',
                      ],
                      train_n_small_int=None,
                      validation_set=False,
                      using_text_with_linebreaks=False,
                      test_in_discharge=False,
                      test_in_inhouse=False,
                      selective_ann_pool=None):
    """Main function"""
    assert document_source in PPS.document_sources, \
        f"document_source must be one of PPS.document_sources ({PPS.document_sources}!)"
    if test_in_discharge:
        assert document_source=='inhouse', \
            "document_source must be inhouse when setting test_in_discharge as True!"
    if test_in_inhouse:
        assert document_source=='discharge_summary', \
            "document_source must be discharge_summary when setting test_in_inhouse as True!"

    #set test_data and train_data
    train_data, valid_data, test_data = \
        import_train_valid_test_data_by_source(document_source=document_source)
    total_data = train_data + valid_data + test_data
    #set df_annotated_data and add text_raw to data_i
    if document_source=='inhouse' and using_text_with_linebreaks:
        df_annotated_data = import_df_annotated_data()
        for t in train_data:
            t['text_raw'] = df_annotated_data.iloc[t['document_id']]['서식내용']
        for t in valid_data:
            t['text_raw'] = df_annotated_data.iloc[t['document_id']]['서식내용']
        for t in test_data:
            t['text_raw'] = df_annotated_data.iloc[t['document_id']]['서식내용']
        total_data = train_data + valid_data + test_data

    print(f"Total sample number: {len(total_data)}")
    if document_source=='inhouse':
        #set train-test data
        if subgroup_category:
            #set document_ids
            document_ids_for_subgroup = DOC_INFOS[subgroup_category][subgroup_from_i]
            document_ids_for_subgroup_test = DOC_INFOS[subgroup_category][subgroup_to_i]
            #set train_data, test_data
            test_data = [td for td in total_data
                            if td['document_id'] in document_ids_for_subgroup_test]
            test_data = random.sample(test_data, min(TEST_SAMPLE_N, len(test_data)))
            train_data = [td for td in total_data
                                    if td['document_id'] in document_ids_for_subgroup]
        else:
            test_data = test_data[:test_sampel_size] if test_sampel_size else test_data
        if test_in_discharge:
            _, _, test_data3 = \
                import_train_valid_test_data_by_source(document_source='discharge_summary')
            test_data = test_data3
    else: #discharge_summary
        print(f"document_source: {document_source}")
        train_data = train_data + valid_data
        if test_in_inhouse:
            _, _, test_data2 = \
                import_train_valid_test_data_by_source(document_source='inhouse')
            test_data = test_data2
    #convert_negation_labels_for_check
    _ = convert_negation_labels_for_check(total_data)

    #for subgroups + samples_n_from
    if subgroup_category:
        print(f" - subgroup_category: {subgroup_category}")
        print(f" - subgroup_from_i: {subgroup_from_i}")
        print(f" - train_n_small_int: {train_n_small_int}")
        print(f"   - subgroup_to_i: {subgroup_to_i}")
        #set generator for individual subgroup_from and subgroup_to
        generator = NERPromptGenerator(test_data, train_data,
                                       embedding_model=PPS.embedding_model,
                                       subgroup_category=subgroup_category,
                                       subgroup_from=subgroup_from_i,
                                       subgroup_to=subgroup_to_i,
                                       train_n_small_int=train_n_small_int,
                                       use_gpu=True)
        generator.generate_and_save_all_combinations(
            output_types_ner=PPS.output_types_ner,
            demonstration_selection_methods=PPS.demonstration_selection_methods,
            demonstration_sample_ns=PPS.demonstration_sample_ns,
            demonstration_sorting_methods=PPS.demonstration_sorting_methods,
            samples_n_from=samples_n_from,
            date_ver=date_ver,
            using_text_with_linebreaks=using_text_with_linebreaks,
            document_source=document_source)
    else: # if subgroup_category is None:
        if selective_ann_pool:
            dir_selected_indices = \
                r'E:\NRF_CCADD\DATASET\240705\train_valid_test_data\selected_indices'
            for lambda_ in selective_ann_pool['lambda']:
                for sample_n_from in samples_n_from:
                    alphas = selective_ann_pool.get('alpha', None)
                    samples_n_from_i = [sample_n_from]
                    lambda_str = f"{lambda_:.1f}".replace('.', '')
                    fname_selected_indices = \
                        f'selected_indices_submodular_k{sample_n_from}_lambda{lambda_str}.pkl'
                    import pickle
                    with open(f"{dir_selected_indices}/{fname_selected_indices}", 'rb') as f:
                        selective_indices = pickle.load(f)
                    train_data_selected = [train_data[i] for i in selective_indices]
                    #default entity_labels
                    generator = NERPromptGenerator(test_data, train_data_selected,
                                                   embedding_model=PPS.embedding_model,
                                                   use_gpu=True)
                    print(f"output_types_ner: {PPS.output_types_ner}")
                    print(f"demonstration_selection_methods: {PPS.demonstration_selection_methods}")
                    print(f"demonstration_sample_ns: {PPS.demonstration_sample_ns}")
                    print(f"entity_labels: {PPS.entity_labels}")
                    if alphas:
                        for alpha in alphas:
                            astr = f"{alpha:.1f}".replace('.', '')
                            fname_selected_indices = \
                                fname_selected_indices.replace('submodular_',
                                                               f'submodular_alpha{astr}.pkl')
                            generator.generate_and_save_all_combinations(
                                PPS.output_types_ner,
                                PPS.demonstration_selection_methods,
                                PPS.demonstration_sample_ns,
                                PPS.demonstration_sorting_methods,
                                date_ver=date_ver,
                                using_text_with_linebreaks=using_text_with_linebreaks,
                                document_source=document_source,
                                samples_n_from=samples_n_from_i,
                                lambda_=lambda_,
                                alpha=alpha,
                                ann_pool_method=selective_ann_pool['method'])
                    
                    else:
                        generator.generate_and_save_all_combinations(
                            PPS.output_types_ner,
                            PPS.demonstration_selection_methods,
                            PPS.demonstration_sample_ns,
                            PPS.demonstration_sorting_methods,
                            date_ver=date_ver,
                            using_text_with_linebreaks=using_text_with_linebreaks,
                            document_source=document_source,
                            samples_n_from=samples_n_from_i,
                            lambda_=lambda_,
                            alpha=alpha,
                            ann_pool_method=selective_ann_pool['method'])
        else: # if selective_ann_pool is None:
            #default entity_labels
            generator = NERPromptGenerator(test_data, train_data,
                                        embedding_model=PPS.embedding_model,
                                        use_gpu=True)
            print(f"output_types_ner: {PPS.output_types_ner}")
            print(f"demonstration_selection_methods: {PPS.demonstration_selection_methods}")
            print(f"demonstration_sample_ns: {PPS.demonstration_sample_ns}")
            print(f"entity_labels: {PPS.entity_labels}")
            generator.generate_and_save_all_combinations(PPS.output_types_ner,
                                                        PPS.demonstration_selection_methods,
                                                        PPS.demonstration_sample_ns,
                                                        PPS.demonstration_sorting_methods,
                                                        date_ver=date_ver,
                                                        using_text_with_linebreaks=\
                                                            using_text_with_linebreaks,
                                                        document_source=document_source,
                                                        samples_n_from=samples_n_from)

            #add_patient_infos - True
            if add_patient_infos:
                print(f"add_patient_infos: {add_patient_infos}")
                generator.generate_and_save_all_combinations(PPS.output_types_ner,
                                                            PPS.demonstration_selection_methods,
                                                            PPS.demonstration_sample_ns,
                                                            PPS.demonstration_sorting_methods,
                                                            date_ver=date_ver,
                                                            add_patient_infos=add_patient_infos,
                                                            patient_infos_categories=\
                                                                patient_infos_categories,
                                                            using_text_with_linebreaks=\
                                                                using_text_with_linebreaks,
                                                            document_source=document_source)

            #using masked text
            if using_masked_text:
                print(f"using_masked_text: {using_masked_text}")
                generator = NERPromptGenerator(test_data, train_data,
                                            embedding_model=PPS.embedding_model,
                                            use_gpu=True,
                                            using_masked_text=True)
                print(f"output_types_ner: {PPS.output_types_ner}")
                print(f"demonstration_selection_methods: {PPS.demonstration_selection_methods}")
                print(f"output_types_ner: {PPS.output_types_ner}")
                print(f"demonstration_sample_ns: {PPS.demonstration_sample_ns}")
                print(f"entity_labels: {PPS.entity_labels}")
                generator.generate_and_save_all_combinations(PPS.output_types_ner,
                                                            PPS.demonstration_selection_methods,
                                                            PPS.demonstration_sample_ns,
                                                            PPS.demonstration_sorting_methods,
                                                            date_ver=date_ver,
                                                            using_text_with_linebreaks=\
                                                                using_text_with_linebreaks,
                                                            document_source=document_source)

            #entity_group - True
            if entity_group:
                print(f"entity_group: {entity_group}")
                print(f"output_types_ner: {PPS.output_types_ner}")
                print(f"demonstration_selection_methods: {PPS.demonstration_selection_methods}")
                print(f"output_types_ner: {PPS.output_types_ner}")
                print(f"demonstration_sample_ns: {PPS.demonstration_sample_ns}")
                print(f"entity_labels: {PPS.entity_labels}")
                generator.generate_and_save_all_combinations(PPS.output_types_ner,
                                                            PPS.demonstration_selection_methods,
                                                            PPS.demonstration_sample_ns,
                                                            PPS.demonstration_sorting_methods,
                                                            entity_group=True,
                                                            date_ver=date_ver,
                                                            using_text_with_linebreaks=\
                                                                using_text_with_linebreaks,
                                                            document_source=document_source)

            #sentence_entity - True
            if sentence_entity:
                print(f"sentence_entity: {sentence_entity}")
                test_data_with_sentence_entity = []
                for test_data_i in test_data:
                    entity_labels_i = set(e['label'] for e in test_data_i['entities'])
                    if any(sentence_entity_label in entity_labels_i
                        for sentence_entity_label in PPS.sentence_entity_labels):
                        test_data_with_sentence_entity.append(test_data_i)
                train_data_with_sentence_entity = []
                for train_data_i in train_data:
                    entity_labels_i = set(e['label'] for e in train_data_i['entities'])
                    if any(sentence_entity_label in entity_labels_i
                        for sentence_entity_label in PPS.sentence_entity_labels):
                        train_data_with_sentence_entity.append(train_data_i)
                generator = NERPromptGenerator(test_data_with_sentence_entity,
                                            train_data_with_sentence_entity,
                                            embedding_model=PPS.embedding_model,
                                            use_gpu=True)
                print(f"test_data_with_sentence_entity: {len(test_data_with_sentence_entity)}")
                print(f"train_data_with_sentence_entity: {len(train_data_with_sentence_entity)}")
                print(f"output_types_ner: {PPS.output_types_ner}")
                print(f"demonstration_selection_methods: {PPS.demonstration_selection_methods}")
                print(f"output_types_ner: {PPS.output_types_ner}")
                print(f"demonstration_sample_ns: {PPS.demonstration_sample_ns}")
                print(f"entity_labels: {PPS.entity_labels}")
                generator.generate_and_save_all_combinations(PPS.output_types_ner,
                                                            PPS.demonstration_selection_methods,
                                                            PPS.demonstration_sample_ns,
                                                            PPS.demonstration_sorting_methods,
                                                            sentence_entity=True,
                                                            date_ver=date_ver,
                                                            using_text_with_linebreaks=\
                                                                using_text_with_linebreaks,
                                                            document_source=document_source)

            #validation_set - True
            if validation_set:
                print(f"validation_set: {validation_set}")
                generator = NERPromptGenerator(valid_data, train_data,
                                            embedding_model=PPS.embedding_model,
                                            use_gpu=True)
                print(f"output_types_ner: {PPS.output_types_ner}")
                print(f"demonstration_selection_methods: {PPS.demonstration_selection_methods}")
                print(f"output_types_ner: {PPS.output_types_ner}")
                print(f"demonstration_sample_ns: {PPS.demonstration_sample_ns}")
                print(f"entity_labels: {PPS.entity_labels}")
                generator.generate_and_save_all_combinations(PPS.output_types_ner,
                                                            PPS.demonstration_selection_methods,
                                                            PPS.demonstration_sample_ns,
                                                            PPS.demonstration_sorting_methods,
                                                            date_ver=date_ver,
                                                            validation_data=True,
                                                            using_text_with_linebreaks=\
                                                                using_text_with_linebreaks,
                                                            document_source=document_source)

# if __name__ == "__main__":
#     set_final_prompts(using_masked_text=True)

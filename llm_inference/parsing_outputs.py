import re
import ast

from settings import PPS
from utils import format_output_ner


def extract_annotated_entities_from_tagged_text(input_text,
                                                matches_tag_and_text,
                                                with_abbr,
                                                entity_labels):
    """extract_annotated_entities"""
    #set abbreviations
    tag2label = {v: k for k, v in PPS.abbreviations.items()} \
        if with_abbr else {el.replace(' ', '-'): el for el in entity_labels}
    # Extract tagged entities
    entities = []
    for tag, text in matches_tag_and_text:
        label = tag2label[tag]
        search_start = 0
        while True:
            start = input_text.find(text, search_start)
            if start == -1:
                break
            end = start + len(text)
            entities.append({
                'text': text,
                'label': label,
                'start': start,
                'end': end
            })
            search_start = end

    return entities

def parse_llm_response(llm_response, text, output_type_ner, entity_labels,
                       entity_group=False, sentence_entity=False):
    """parse_llm_response"""
    def find_matching_category(label_str, entity_labels):
        """find_matching_category"""
        for entity_label in entity_labels:
            if label_str.startswith(entity_label):
                return entity_label
        raise ValueError("No matching category found for the input string")
    def find_matches_tag_and_text(line):
        """find_tagged_text_line"""
        pattern = r'<([\w-]+)>(.*?)</\1>'
        matches = re.finditer(pattern, line)
        matches_tag_and_text = [(match.group(1), match.group(2)) for match in matches]
        return matches_tag_and_text

    #parse the response
    entities = []
    current_label = None
    if output_type_ner=='entity_list_with_start_end':
        if entity_group is False and sentence_entity is False:
            llm_response = "### Confirmed Diagnosis (CD): \n" + llm_response
        for line in llm_response.split('\n'):
            line = line.strip()
            if line.startswith('*'):
                try:
                    parts = line[1:].strip().split('(start')
                    text_entity = parts[0].strip().replace('"', '').strip()
                    info = parts[1].split(')')[0].split(',')
                    start = int(info[0].split(':')[1].strip())
                    end = int(info[1].split(':')[1].strip())
                    current_entity = {
                        'text': text_entity,
                        'start': start,
                        'end': end,
                        'label': current_label
                    }
                    entities.append(current_entity)
                except (ValueError, IndexError):
                    continue
            if line.startswith('###') and any(k in line for k in entity_labels):
                current_label = [k for k in entity_labels if k in line]
                assert len(current_label)==1, \
                    "Multiple entity labels found in ### line for entity_list_with_start_end"
                current_label = current_label[0]
        # Remove entities with None label
        entities = [entity for entity in entities if entity['label'] is not None]

    elif output_type_ner=='entity_list_appearance_order_with_entity_label':
        for line in llm_response.split('\n'):
            line = line.strip()
            if line.startswith('*'):
                try:
                    parts = line[1:].strip().split('(entity label:')
                    text_entity = parts[0].strip().replace('"', '').strip()
                    label_str = parts[1].split(')')[0].strip()
                    #check whether label is valid (for entity_labels)
                    label = find_matching_category(label_str, entity_labels)
                    current_entity = {
                        'text': text_entity,
                        'label': label
                    }
                    entities.append(current_entity)
                except (ValueError, IndexError):
                    continue

    elif output_type_ner=='entity_list_appearance_order_json_format':
        for line in llm_response.split('\n'):
            line = line.strip()
            if line.startswith('[{'):
                entities = ast.literal_eval(line.strip())

    elif output_type_ner=='tagged_text_with_abbreviations':
        for line in llm_response.split('\n'):
            line = line.strip()
            matches_tag_and_text = find_matches_tag_and_text(line)
            if matches_tag_and_text:
                entities_in_a_line = \
                    extract_annotated_entities_from_tagged_text(text,
                                                                matches_tag_and_text,
                                                                with_abbr=True,
                                                                entity_labels=entity_labels)
                entities.extend(entities_in_a_line)

    elif output_type_ner=='tagged_text_without_abbrevaitions':
        for line in llm_response.split('\n'):
            line = line.strip()
            matches_tag_and_text = find_matches_tag_and_text(line)
            if matches_tag_and_text:
                entities_in_a_line = \
                    extract_annotated_entities_from_tagged_text(text,
                                                                matches_tag_and_text,
                                                                with_abbr=False,
                                                                entity_labels=entity_labels)
                entities.extend(entities_in_a_line)

    else:
        assert False, f"Unknown output_type ({output_type_ner})!"

    #check keys of entities
    for entity in entities:
        _ = entity['text']
        _ = entity['label']

    return entities

def make_tokens_readable(tokens):
    """make_tokens_readable"""
    readable_tokens = []
    for token in tokens:
        # Replace 'Ġ' with a space, unless it's at the start of the token
        if token.startswith('Ġ'):
            token = '#' + token[1:]
        # Replace other special characters
        token = token.replace('Ċ', '@').replace('ĊĊ', '@@')
        token = token.replace('Ġ', '##')  # Remove any remaining 'Ġ'
        readable_tokens.append(token)
    return readable_tokens

def get_entities_in_output_format(output_type_ner,
                                  condition,
                                  text,
                                  entities):
    """get_entities_in_output_format"""
    if condition['sentence_entity']:
        entity_labels = PPS.sentence_entity_labels
    elif condition['entity_group']:
        entity_labels = list(PPS.entity_groups.keys()) \
            + ['Time Information', 'Negation']
    else:
        entity_labels = PPS.entity_labels
    output_format_ner = \
        PPS.output_formats_ner[output_type_ner]
    formatted_output_ner = format_output_ner(output_type_ner,
                                             entity_labels,
                                             entities,
                                             text)
    if 'entity_list' in output_type_ner:
        entities_in_output_format = output_format_ner.format(
            document_type_str='clinical notes',
            input_text=text,
            entity_list=formatted_output_ner
        )
    else:
        entities_in_output_format = output_format_ner.format(
            document_type_str='clinical notes',
            input_text=text,
            tagged_text=formatted_output_ner
        )
    return entities_in_output_format

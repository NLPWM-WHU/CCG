import os
from random import shuffle
from transformers import AutoTokenizer
from file_io import read_tsv_file, write_tsv_file, read_json_file, write_json_file


def find_max_length_of_tokens(folder, dataset_file, model_path="gpt2-base"):
    dataset_path = os.path.join(folder, dataset_file)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    instances = read_tsv_file(dataset_path)
    max_length = 0

    for instance in instances:
        text = instance[1]
        tokens = tokenizer.encode(text, return_tensors='pt')[0]
        if len(tokens) > max_length:
            max_length = len(tokens)

    return max_length


def sort_dict_by_value(unsorted_dict):
    sorted_tuples = sorted(unsorted_dict.items(), key=lambda x: x[1], reverse=True)

    sorted_dict = {}
    for sorted_tuple in sorted_tuples:
        sorted_dict[sorted_tuple[0]] = sorted_tuple[1]

    return sorted_dict


def construct_relationword(
        folder,
        dataset_file,
        triggerword_file="triggerword.json",
        relationword_file="relationword.json"):

    dataset_path = os.path.join(folder, dataset_file)
    triggerword_path = os.path.join(folder, triggerword_file)
    relationword_path = os.path.join(folder, relationword_file)

    dataset = read_json_file(dataset_path)
    triggerword = read_json_file(triggerword_path)
    relationword = {}

    for instance in dataset:
        instance_id = instance["id"]
        relation = instance["relation"]
        token = instance["token"]

        if instance_id in triggerword:
            trigger_words = triggerword[instance_id]
            trigger_words = " ".join([token[int(trigger_word)] for trigger_word in trigger_words])
            if relation not in relationword:
                relationword[relation] = {trigger_words: 1}
            else:
                if trigger_words not in relationword[relation]:
                    relationword[relation][trigger_words] = 1
                else:
                    relationword[relation][trigger_words] += 1

    for relation in relationword:
        relation_words = []
        for relation_word in sort_dict_by_value(relationword[relation]):
            relation_words.append(relation_word)
        relationword[relation] = relation_words

    write_json_file(relationword_path, relationword)


def transfer_json_2_tsv(folder, input_file, output_file):
    input_file_path = os.path.join(folder, input_file)
    output_file_path = os.path.join(folder, output_file)

    input_instances = read_json_file(input_file_path)
    output_instances = []

    for instance in input_instances:
        tokens = instance["token"]
        relation = instance["relation"]
        entity1_start = instance["subj_start"]
        entity1_end = instance["subj_end"]
        entity2_start = instance["obj_start"]
        entity2_end = instance["obj_end"]

        tokens[entity1_start] = "<e1> " + tokens[entity1_start]
        tokens[entity1_end] = tokens[entity1_end] + " </e1>"
        tokens[entity2_start] = "<e2> " + tokens[entity2_start]
        tokens[entity2_end] = tokens[entity2_end] + " </e2>"

        sentence = " ".join(tokens)
        output_instances.append([relation, sentence])
        write_tsv_file(output_file_path, output_instances)


def shuffle_instances(folder, input_file, output_file):
    instances = read_tsv_file(os.path.join(folder, input_file))
    shuffle(instances)
    write_tsv_file(os.path.join(folder, output_file), instances)


if __name__ == '__main__':
    print(find_max_length_of_tokens("ace2005", "train.tsv"))
    # construct_relationword("1.0", "train-1.0.json")
    # transfer_json_2_tsv("1.0", "train-1.0.json", "train-1.0.tsv")
    # shuffle_instances("32-shot", "inference_mlm_aug.tsv", "inference_mlm_aug.tsv")

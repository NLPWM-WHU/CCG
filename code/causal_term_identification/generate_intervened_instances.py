import tqdm
from file_io import *
from transformers import pipeline


def substitute_context_word(instance, span_index, span=100):
    unmasker = pipeline("fill-mask", model="../bert-base", top_k=span_index*span)

    instance_id = instance["id"]
    token = instance["token"]
    relation = instance["relation"]
    entity1_start = instance["subj_start"]
    entity1_end = instance["subj_end"]
    entity2_start = instance["obj_start"]
    entity2_end = instance["obj_end"]

    instance_substitutions = []
    for index in range(entity1_end + 1, entity2_start):
        masked_token = token[:index] + ["[MASK]"] + token[index + 1:]
        masked_sentence = " ".join(masked_token)
        unmasked_results = unmasker(masked_sentence)
        for unmasked_result in unmasked_results[(span_index-1)*span: span_index*span]:
            unmasked_word = unmasked_result["token_str"]
            unmasked_token = token[:entity1_start] + ["<e1>"] + token[entity1_start: entity1_end + 1] + ["</e1>"] + \
                             token[entity1_end + 1: index] + [unmasked_word] + token[index + 1: entity2_start] + \
                             ["<e2>"] + token[entity2_start: entity2_end + 1] + ["</e2>"] + token[entity2_end + 1:]
            unmasked_sentence = " ".join(unmasked_token)
            instance_substitutions.append([relation, unmasked_sentence, instance_id, index])
    return instance_substitutions


def delete_context_word(instance):
    instance_id = instance["id"]
    token = instance["token"]
    relation = instance["relation"]
    entity1_start = instance["subj_start"]
    entity1_end = instance["subj_end"]
    entity2_start = instance["obj_start"]
    entity2_end = instance["obj_end"]

    instance_deletions = []
    for index in range(entity1_end + 1, entity2_start):
        deleted_token = token[:entity1_start] + ["<e1>"] + token[entity1_start: entity1_end + 1] + ["</e1>"] + \
                             token[entity1_end + 1: index] + token[index + 1: entity2_start] + \
                             ["<e2>"] + token[entity2_start: entity2_end + 1] + ["</e2>"] + token[entity2_end + 1:]
        deleted_sentence = " ".join(deleted_token)
        instance_deletions.append([relation, deleted_sentence, instance_id, index])
    return instance_deletions


def main():
    dataset_file = "ace2005/wl.json"
    intervened_file = "ace2005/substitution.tsv"

    instances = read_json_file(dataset_file)
    intervened_instances = []
    for instance in tqdm.tqdm(instances):
        intervened_instances.extend(substitute_context_word(instance, 1))

    write_tsv_file(intervened_file, intervened_instances)


if __name__ == '__main__':
    main()

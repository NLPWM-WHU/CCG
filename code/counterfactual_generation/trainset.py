import os
from file_io import read_json_file, write_tsv_file


relation_to_instruction = {
    "Message-Topic(e1,e2)": "message-topic",
    "Message-Topic(e2,e1)": "topic-message",
    "Entity-Destination(e1,e2)": "entity-destination",
    "Entity-Destination(e2,e1)": "destination-entity",
    "Content-Container(e1,e2)": "content-container",
    "Content-Container(e2,e1)": "container-content",
    "Cause-Effect(e2,e1)": "effect-cause",
    "Cause-Effect(e1,e2)": "cause-effect",
    "Component-Whole(e2,e1)": "whole-component",
    "Component-Whole(e1,e2)": "component-whole",
    "Member-Collection(e2,e1)": "collection-member",
    "Member-Collection(e1,e2)": "member-collection",
    "Instrument-Agency(e2,e1)": "agency-instrument",
    "Instrument-Agency(e1,e2)": "instrument-agency",
    "Product-Producer(e2,e1)": "producer-product",
    "Product-Producer(e1,e2)": "product-producer",
    "Entity-Origin(e1,e2)": "entity-origin",
    "Entity-Origin(e2,e1)": "origin-entity"
}


relation_to_instruction_ace2005 = {
    "PER-SOC": "per-soc",
    "ART": "art",
    "PHYS": "phys",
    "ORG-AFF": "org-aff",
    "PART-WHOLE": "part-whole",
    "GEN-AFF": "gen-aff"
}


def generate_trainset_old_version(
        folder,
        dataset_file,
        triggerword_file="triggerword.json",
        trainset_file="train.tsv",
        excluded_relation="Other"):

    dataset_path = os.path.join(folder, dataset_file)
    triggerword_path = os.path.join(folder, triggerword_file)
    trainset_path = os.path.join(folder, trainset_file)

    dataset = read_json_file(dataset_path)
    triggerword = read_json_file(triggerword_path)
    trainset = []

    for instance in dataset:
        instance_id = instance["id"]
        relation = instance["relation"]

        if relation != excluded_relation and instance_id in triggerword:
            token = instance["token"]
            entity1_start = instance["subj_start"]
            entity2_end = instance["obj_end"]

            masked_token = token[:]
            trigger_words = triggerword[instance_id]
            for trigger_word in trigger_words:
                masked_token[int(trigger_word)] = "<mask>"

            instruction = relation_to_instruction[relation]
            insert_index = int(trigger_words[0]) - 1
            masked_token[insert_index] += " (" + instruction + ")"
            token[insert_index] += " (" + instruction + ")"

            truncated_masked_token = masked_token[entity1_start: entity2_end + 1]
            truncated_token = token[entity1_start: entity2_end + 1]
            prompt = ["sentence1:"] + truncated_masked_token + ["sentence2:"] + truncated_token
            prompt = " ".join(prompt)
            trainset.append([relation, prompt])

    write_tsv_file(trainset_path, trainset)


def generate_trainset(
        folder,
        dataset_file,
        triggerword_file="triggerword.json",
        trainset_file="train.tsv",
        excluded_relation="Other"):

    dataset_path = os.path.join(folder, dataset_file)
    triggerword_path = os.path.join(folder, triggerword_file)
    trainset_path = os.path.join(folder, trainset_file)

    dataset = read_json_file(dataset_path)
    triggerword = read_json_file(triggerword_path)
    trainset = []

    for instance in dataset:
        instance_id = instance["id"]
        relation = instance["relation"]

        if relation != excluded_relation and instance_id in triggerword:
            token = instance["token"]
            entity1_start = instance["subj_start"]
            entity1_end = instance["subj_end"]
            entity2_start = instance["obj_start"]
            entity2_end = instance["obj_end"]
            entity1 = " ".join(token[entity1_start: entity1_end + 1])
            entity2 = " ".join(token[entity2_start: entity2_end + 1])

            instruction = relation_to_instruction[relation]
            instruction = "(" + instruction + ")"

            masked_span = "<mask>"
            trigger_words = triggerword[instance_id]
            trigger_words = " ".join([token[int(trigger_word)] for trigger_word in trigger_words])

            prompt_part1 = ["sentence1:"] + [entity1] + [instruction] + [masked_span] + [entity2]
            prompt_part2 = ["sentence2:"] + [entity1] + [instruction] + [trigger_words] + [entity2]
            prompt = prompt_part1 + prompt_part2
            prompt = " ".join(prompt)
            trainset.append([relation, prompt])

    write_tsv_file(trainset_path, trainset)


if __name__ == '__main__':
    generate_trainset("1.0", "train-1.0.json")

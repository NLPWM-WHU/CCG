import os
import tqdm
import torch
from nltk.corpus import wordnet
from stanfordcorenlp import StanfordCoreNLP
from file_io import read_json_file, write_tsv_file
from transformers import GPT2LMHeadModel, AutoTokenizer


corenlp_parser = StanfordCoreNLP("stanford-corenlp-4.5.1")


relation_to_instruction = {
    "PER-SOC": "per-soc",
    "ART": "art",
    "PHYS": "phys",
    "ORG-AFF": "org-aff",
    "PART-WHOLE": "part-whole",
    "GEN-AFF": "gen-aff"
}


def gpt_generate(input_prompt, tokenizer, model, device):
    prefix_ids = tokenizer.encode(input_prompt, return_tensors='pt').to(device)
    outputs = model.generate(prefix_ids, max_length=200, num_beams=100, num_return_sequences=10, early_stopping=True)
    output_prompts = []
    for output in outputs:
        output_prompt = tokenizer.decode(output, skip_special_tokens=True)
        output_prompts.append(output_prompt)
    return output_prompts


def remove_beverb(input_sequence):
    beverb = ["be", "am", "is", "are", "been", "being", "was", "were"]

    output_sequence = []
    for word in input_sequence.split(" "):
        if word not in beverb:
            output_sequence.append(word)

    output_sequence = " ".join(output_sequence)
    return output_sequence


def remove_determiner(input_sequence):
    pos_tags = corenlp_parser.pos_tag(input_sequence)
    output_sequence = []

    for pos_tag in pos_tags:
        word = pos_tag[0]
        pos_tag = pos_tag[-1]
        if pos_tag != "DT":
            output_sequence.append(word)

    output_sequence = " ".join(output_sequence)
    return output_sequence


def remove_tense_of_word(tense_words):
    tense_words = tense_words.split(" ")
    no_tense_words = []

    for tense_word in tense_words:
        no_tense_word = wordnet.morphy(tense_word)
        if no_tense_word:
            no_tense_words.append(no_tense_word)
        else:
            no_tense_words.append(tense_word)

    no_tense_words = " ".join(no_tense_words)
    return no_tense_words


def judge_relation_match(trigger_words, relation_words):
    match_flag = False
    trigger_words = remove_beverb(trigger_words)
    trigger_words = remove_determiner(trigger_words)

    for relation_word in relation_words:
        relation_word = remove_beverb(relation_word)
        relation_word = remove_determiner(relation_word)
        if remove_tense_of_word(relation_word) == remove_tense_of_word(trigger_words):
            if remove_tense_of_word(trigger_words) != "":
                match_flag = True
                break

    return match_flag


def conduct_new_relation_constrain(new_relations, new_relation_constrain):
    if len(new_relations) <= new_relation_constrain:
        return new_relations
    else:
        return new_relations[:new_relation_constrain]


def infer_by_trained_gpt(
        folder,
        dataset_file,
        new_relation_constrain,
        triggerword_file="triggerword.json",
        newrelation_file="newrelation.json",
        relationword_file="relationword.json",
        inference_file="inference.tsv",
        gpt_path="saved_model"):

    dataset_path = os.path.join(folder, dataset_file)
    triggerword_path = os.path.join(folder, triggerword_file)
    newrelation_path = os.path.join(folder, newrelation_file)
    relationword_path = os.path.join(folder, relationword_file)
    inference_path = os.path.join(folder, inference_file)

    dataset = read_json_file(dataset_path)
    triggerword = read_json_file(triggerword_path)
    newrelation = read_json_file(newrelation_path)
    relationword = read_json_file(relationword_path)
    inference = []

    tokenizer = AutoTokenizer.from_pretrained(gpt_path)
    model = GPT2LMHeadModel.from_pretrained(gpt_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for instance in tqdm.tqdm(dataset):
        instance_id = instance["id"]
        relation = instance["relation"]
        token = instance["token"]
        entity1_start = instance["subj_start"]
        entity1_end = instance["subj_end"]
        entity2_start = instance["obj_start"]
        entity2_end = instance["obj_end"]

        if instance_id in newrelation and instance_id in triggerword:
            entity1 = " ".join(token[entity1_start: entity1_end + 1])
            entity2 = " ".join(token[entity2_start: entity2_end + 1])
            token[entity1_start] = "<e1> " + token[entity1_start]
            token[entity1_end] = token[entity1_end] + " </e1>"
            token[entity2_start] = "<e2> " + token[entity2_start]
            token[entity2_end] = token[entity2_end] + " </e2>"

            trigger_words = triggerword[instance_id]
            insert_position = int(trigger_words[0])

            trigger_words.reverse()
            for trigger_word in trigger_words:
                del token[int(trigger_word)]
            under_inserted_token = token

            new_relations = newrelation[instance_id]
            for new_relation in conduct_new_relation_constrain(new_relations, new_relation_constrain):
                instruction = "(" + relation_to_instruction[new_relation] + ")"
                input_prompt = ["sentence1:"] + [entity1] + [instruction] + ["<mask>"] + [entity2] + ["sentence2:"]
                input_prompt = " ".join(input_prompt)
                output_prompts = gpt_generate(input_prompt, tokenizer, model, device)

                for output_prompt in output_prompts:
                    sentence2 = output_prompt.split("sentence2:")[-1].strip()
                    sentence2_wo_entity1 = sentence2.replace(entity1, "").strip()
                    sentence2_wo_instruction = sentence2_wo_entity1.replace(instruction, "").strip()
                    generated_trigger_words = sentence2_wo_instruction.replace(entity2, "").strip()

                    if generated_trigger_words:
                        old_relation_words = relationword[relation]
                        new_relation_words = relationword[new_relation]
                        old_relation_match = judge_relation_match(generated_trigger_words, old_relation_words)
                        new_relation_match = judge_relation_match(generated_trigger_words, new_relation_words)

                        other_relations = []
                        for relation in relation_to_instruction:
                            if relation != new_relation:
                                other_relations.append(relation)

                        other_relation_match = False
                        for other_relation in other_relations:
                            other_relation_words = relationword[other_relation]
                            other_relation_match = judge_relation_match(generated_trigger_words, other_relation_words)
                            if other_relation_match:
                                other_relation_match = True
                                break

                        if new_relation_match:
                            if not old_relation_match and not other_relation_match:
                                new_token = under_inserted_token[:]
                                new_token.insert(insert_position, generated_trigger_words)
                                new_text = " ".join(new_token)
                                inference.append([new_relation, new_text])
                                break

    write_tsv_file(inference_path, inference)


if __name__ == '__main__':
    infer_by_trained_gpt("ace2005", "train-wl.json", 1)

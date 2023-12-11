from file_io import *
from stanfordcorenlp import StanfordCoreNLP


corenlp_parser = StanfordCoreNLP("stanford-corenlp-4.5.1")


def judge_non_branch(branch_content):
    pure_branch_content = ""
    pure_flag = False
    for char in branch_content:
        if char == " " and not pure_flag:
            continue
        elif char == "(" and not pure_flag:
            pure_flag = True
            pure_branch_content += char
        else:
            pure_branch_content += char

    judge_flag = False
    for char in pure_branch_content:
        if judge_flag:
            if char == "(":
                return False
            else:
                return True

        if char == " ":
            judge_flag = True

    return False


def adjust_constituency_tree(constituency_tree):
    constituency_tree = constituency_tree.split("\n")
    constituency_tree.reverse()

    immigrant_branch_content = ""
    adjusted_constituency_tree = []

    for branch_content in constituency_tree:
        if immigrant_branch_content:
            branch_content += " " + immigrant_branch_content
            immigrant_branch_content = ""

        if judge_non_branch(branch_content):
            pure_branch_content = ""
            pure_flag = False
            for char in branch_content:
                if char == " " and not pure_flag:
                    continue
                elif char == "(" and not pure_flag:
                    pure_flag = True
                    pure_branch_content += char
                else:
                    pure_branch_content += char

            immigrant_branch_content = pure_branch_content
        else:
            adjusted_constituency_tree.append(branch_content)

    adjusted_constituency_tree.reverse()
    adjusted_constituency_tree = "\n".join(adjusted_constituency_tree)
    return adjusted_constituency_tree


def branch_constituency_parse(branch_index, branch_content):
    layer_index = 0
    while branch_content[layer_index] != "(":
        layer_index += 1

    branch_constituency = ""
    for char in branch_content[layer_index + 1:]:
        if char == " ":
            break
        else:
            branch_constituency += char

    branch_words = {}
    if layer_index + 1 + len(branch_constituency) < len(branch_content):
        rest_branch_content = branch_content[layer_index + 1 + len(branch_constituency) + 1:]

        pos_flag = False
        word_flag = False
        pos = ""
        word = ""
        for char in rest_branch_content:
            if char == "(":
                pos_flag = True
            if pos_flag and char == " ":
                pos_flag = False
                word_flag = True
            if word_flag and char == ")":
                word_flag = False
                branch_words[word] = {"position": str(branch_index) + "-" + str(layer_index), "pos": pos,
                                      "constituencies": [branch_constituency + "-" + str(branch_index)
                                                         + "-" + str(layer_index)]}
                pos = ""
                word = ""

            if pos_flag and char != "(":
                pos += char
            if word_flag and char != " ":
                word += char

    branch_constituency = branch_constituency + "-" + str(branch_index) + "-" + str(layer_index)
    branch_constituency = {branch_constituency: list(branch_words.keys())}
    return branch_constituency, branch_words


def constituency_parse(sentence):
    constituency_tree = corenlp_parser.parse(sentence)
    adjusted_constituency_tree = adjust_constituency_tree(constituency_tree)

    tree_constituencies = {}
    tree_words = {}

    for branch_index,  branch_content in enumerate(adjusted_constituency_tree.split("\n")):
        branch_constituency, branch_words = branch_constituency_parse(branch_index, branch_content)
        tree_constituencies.update(branch_constituency)
        tree_words.update(branch_words)

    tree_constituencies_keys = list(tree_constituencies.keys())
    tree_constituencies_keys.reverse()

    for tree_word_key in tree_words.keys():
        tree_word_position = tree_words[tree_word_key]["position"]
        tree_word_branch_index = int(tree_word_position.split("-")[0])
        tree_word_layer_index = int(tree_word_position.split("-")[1])
        current_highest_layer = tree_word_layer_index

        for tree_constituencies_key in tree_constituencies_keys:
            tree_constituency_branch_index = int(tree_constituencies_key.split("-")[-2])
            if tree_constituency_branch_index < tree_word_branch_index:
                tree_constituency_layer_index = int(tree_constituencies_key.split("-")[-1])
                if tree_constituency_layer_index < current_highest_layer:
                    tree_constituencies[tree_constituencies_key].append(tree_word_key)
                    tree_words[tree_word_key]["constituencies"].append(tree_constituencies_key)
                    current_highest_layer = tree_constituency_layer_index

    return tree_constituencies, tree_words


def retrieve_specific_element(input_list, target_element):
    target_index = []

    for index, element in enumerate(input_list):
        if element == target_element:
            target_index.append(index)

    return target_index


def revise_by_constituency_parse(tree_constituencies, tree_words, tokens, preliminary_trigger_words):
    preliminary_trigger_words = [tokens[int(index)] for index in preliminary_trigger_words]
    secondary_trigger_words = []

    for trigger_word in preliminary_trigger_words:
        try:
            trigger_word_constituencies = tree_words[trigger_word]["constituencies"]
        except KeyError:
            continue

        nearest_vp_constituency = None
        for trigger_word_constituency in trigger_word_constituencies:
            if "VP" in trigger_word_constituency:
                nearest_vp_constituency = trigger_word_constituency
                break

        if nearest_vp_constituency is not None:
            constituency_words = tree_constituencies[nearest_vp_constituency]
            for constituency_word in constituency_words:
                if constituency_word not in preliminary_trigger_words:
                    constituency_word_pos = tree_words[constituency_word]["pos"]
                    if "VB" in constituency_word_pos or constituency_word_pos == "IN":
                        retrieved_words = retrieve_specific_element(tokens, constituency_word)
                        for retrieved_word in retrieved_words:
                            if retrieved_word not in secondary_trigger_words:
                                secondary_trigger_words.append(retrieved_word)
        else:
            nearest_pp_constituency = None
            for trigger_word_constituency in trigger_word_constituencies:
                if "PP" in trigger_word_constituency:
                    nearest_pp_constituency = trigger_word_constituency
                    break

            if nearest_pp_constituency is not None:
                constituency_words = tree_constituencies[nearest_pp_constituency]
                for constituency_word in constituency_words:
                    if constituency_word not in preliminary_trigger_words:
                        constituency_word_pos = tree_words[constituency_word]["pos"]
                        if constituency_word_pos == "IN":
                            retrieved_words = retrieve_specific_element(tokens, constituency_word)
                            for retrieved_word in retrieved_words:
                                if retrieved_word not in secondary_trigger_words:
                                    secondary_trigger_words.append(retrieved_word)

    return secondary_trigger_words


def filter_illegal_word(secondary_trigger_words, entity1_end, entity2_start):
    filtered_secondary_trigger_words = []

    for secondary_trigger_word in secondary_trigger_words:
        if entity1_end < secondary_trigger_word < entity2_start:
            filtered_secondary_trigger_words.append(secondary_trigger_word)

    return filtered_secondary_trigger_words


def main():
    dataset_file = "ace2005/wl.json"
    preliminary_trigger_file = "ace2005/preliminary_trigger_words.json"
    revised_trigger_file = "ace2005/revised_trigger_words.json"

    instances = read_json_file(dataset_file)
    preliminary_trigger_words = read_json_file(preliminary_trigger_file)
    revised_trigger_words = preliminary_trigger_words

    for instance in instances:
        instance_id = instance["id"]
        if instance_id in preliminary_trigger_words.keys():
            tokens = instance["token"]
            entity1_start = instance["subj_start"]
            entity1_end = instance["subj_end"]
            entity2_start = instance["obj_start"]
            entity2_end = instance["obj_end"]
            sentence = " ".join(tokens[entity1_start: entity2_end+1])

            tree_constituencies, tree_words = constituency_parse(sentence)
            instance_preliminary_trigger_words = preliminary_trigger_words[instance_id]
            secondary_trigger_words = revise_by_constituency_parse(tree_constituencies, tree_words,
                                                                    tokens, instance_preliminary_trigger_words)
            filtered_secondary_trigger_words = filter_illegal_word(secondary_trigger_words, entity1_end, entity2_start)
            filtered_secondary_trigger_words = [str(index) for index in filtered_secondary_trigger_words]

            if filtered_secondary_trigger_words:
                instance_revised_trigger_words = instance_preliminary_trigger_words
                instance_revised_trigger_words.extend(filtered_secondary_trigger_words)
                revised_trigger_words[instance_id] = instance_revised_trigger_words

    for instance_id in revised_trigger_words:
        instance_trigger_words = revised_trigger_words[instance_id]
        instance_trigger_words = [int(index) for index in instance_trigger_words]
        instance_trigger_words.sort()
        instance_trigger_words = [str(index) for index in instance_trigger_words]
        revised_trigger_words[instance_id] = instance_trigger_words

    write_json_file(revised_trigger_file, revised_trigger_words)


if __name__ == '__main__':
    main()

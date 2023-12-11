import os
import tqdm
from math import ceil
from file_io import read_json_file, write_json_file
from utils import preprocess_entity, obtain_hypernym_path, sort_dict_by_value


def obtain_step_permutations(current_step, entity1_step_ceiling, entity2_step_ceiling):
    step_permutations = []

    entity1_allowed_step_ceiling = min(current_step, entity1_step_ceiling)
    entity2_allowed_step_ceiling = min(current_step, entity2_step_ceiling)

    for entity1_step in range(entity1_allowed_step_ceiling + 1):
        for entity2_step in range(entity2_allowed_step_ceiling + 1):
            if entity1_step + entity2_step == current_step:
                step_permutations.append([entity1_step, entity2_step])

    return step_permutations


def judge_same_relation(relation, compared_relation):
    if relation == compared_relation:
        return True
    else:
        return False


def jude_same_relation_type(relation, compared_relation, excluded_relation="Other"):
    if compared_relation == excluded_relation:
        return judge_same_relation(relation, compared_relation)
    else:
        relation_type = relation[:-7]
        compared_relation_type = compared_relation[:-7]
        if relation_type == compared_relation_type:
            return True
        else:
            return False


def construct_newrelation(
        folder,
        dataset_file,
        step_constrain,
        relationnet_file="relationnet.json",
        newrelation_file="newrelation.json"):

    dataset_path = os.path.join(folder, dataset_file)
    relationnet_path = os.path.join(folder, relationnet_file)
    newrelation_path = os.path.join(folder, newrelation_file)

    dataset = read_json_file(dataset_path)
    relationnet = read_json_file(relationnet_path)
    newrelation = {}
    counter = 0
    for instance in tqdm.tqdm(dataset):
        instance_id = instance["id"]
        relation = instance["relation"]

        token = instance["token"]
        entity1_index = instance["subj_end"]
        entity2_index = instance["obj_end"]
        entity1 = token[entity1_index]
        entity2 = token[entity2_index]
        entity1 = preprocess_entity(entity1)
        entity2 = preprocess_entity(entity2)
        entity1_hypernym_path = obtain_hypernym_path(entity1)
        entity2_hypernym_path = obtain_hypernym_path(entity2)

        if entity1_hypernym_path and entity2_hypernym_path:
            entity1_step_ceiling = len(entity1_hypernym_path) - 1
            entity2_step_ceiling = len(entity2_hypernym_path) - 1

            new_relations = []
            # new_relation_types = []
            max_step = ceil((entity1_step_ceiling + entity2_step_ceiling) * step_constrain)

            for step in range(max_step + 1):
                step_permutations = obtain_step_permutations(step, entity1_step_ceiling, entity2_step_ceiling)
                step_candidates = {}

                for step_permutation in step_permutations:
                    entity1_step = step_permutation[0]
                    entity2_step = step_permutation[1]
                    entity1_hypernym = entity1_hypernym_path[entity1_step]
                    entity2_hypernym = entity2_hypernym_path[entity2_step]
                    synset_pair = entity1_hypernym.name() + "-" + entity2_hypernym.name()

                    if synset_pair in relationnet:
                        permutation_candidates = relationnet[synset_pair]
                        for candidate in permutation_candidates:
                            if candidate not in step_candidates:
                                step_candidates[candidate] = permutation_candidates[candidate]
                            else:
                                step_candidates[candidate] += permutation_candidates[candidate]

                sorted_step_candidates = sort_dict_by_value(step_candidates)
                for candidate_relation in sorted_step_candidates:
                    if not judge_same_relation(candidate_relation, relation):
                        if candidate_relation not in new_relations:
                            new_relations.append(candidate_relation)
                            # new_relation_types.append(candidate_relation[:-7])

            if new_relations:
                counter += 1
                newrelation[instance_id] = new_relations
    print(counter)
    write_json_file(newrelation_path, newrelation)


if __name__ == '__main__':
    construct_newrelation("1.0", "train-1.0.json", 0.8)

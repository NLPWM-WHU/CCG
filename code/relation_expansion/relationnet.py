import os
import tqdm
from file_io import read_json_file, write_json_file
from utils import preprocess_entity, obtain_hypernym_path, sort_dict_by_value


def sort_relationnet(unsorted_relationnet):
    sorted_relationnet = {}

    for synset_pair in unsorted_relationnet.keys():
        relation_distribution = unsorted_relationnet[synset_pair]
        sorted_relation_distribution = sort_dict_by_value(relation_distribution)
        sorted_relationnet[synset_pair] = sorted_relation_distribution

    return sorted_relationnet


def construct_relationnet(folder, dataset_file, relationnet_file="relationnet.json", excluded_relation=["Other"]):
    dataset_path = os.path.join(folder, dataset_file)
    relationnet_path = os.path.join(folder, relationnet_file)

    dataset = read_json_file(dataset_path)
    relationnet = {}

    for instance in tqdm.tqdm(dataset):
        relation = instance["relation"]
        if relation not in excluded_relation:
            token = instance["token"]
            entity1_index = instance["subj_end"]
            entity2_index = instance["obj_end"]
            entity1 = token[entity1_index]
            entity2 = token[entity2_index]
            preprocessed_entity1 = preprocess_entity(entity1)
            preprocessed_entity2 = preprocess_entity(entity2)
            entity1_hypernym_path = obtain_hypernym_path(preprocessed_entity1)
            entity2_hypernym_path = obtain_hypernym_path(preprocessed_entity2)

            if entity1_hypernym_path and entity2_hypernym_path:
                for entity1_hypernym in entity1_hypernym_path:
                    for entity2_hypernym in entity2_hypernym_path:
                        synset_pair = entity1_hypernym.name() + "-" + entity2_hypernym.name()
                        if synset_pair not in relationnet.keys():
                            relationnet[synset_pair] = {relation: 1}
                        else:
                            if relation not in relationnet[synset_pair].keys():
                                relationnet[synset_pair][relation] = 1
                            else:
                                relationnet[synset_pair][relation] += 1

    sorted_relationnet = sort_relationnet(relationnet)
    write_json_file(relationnet_path, sorted_relationnet)


if __name__ == '__main__':
    construct_relationnet("1.0", "train-1.0.json", excluded_relation=["Other"])

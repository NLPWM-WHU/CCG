from nltk.corpus import wordnet


def preprocess_entity(entity):
    if "-" in entity:
        entity = entity.split("-")[-1]

    entity = entity.lower()
    return entity


def obtain_synsets(word):
    word_root = wordnet.morphy(word)

    if word_root:
        synsets = wordnet.synsets(word_root, pos=wordnet.NOUN)
    else:
        synsets = wordnet.synsets(word, pos=wordnet.NOUN)

    return synsets


def obtain_hypernym_path(word):
    word_root = wordnet.morphy(word)
    if word_root:
        synsets = wordnet.synsets(word_root, pos=wordnet.NOUN)
    else:
        synsets = wordnet.synsets(word, pos=wordnet.NOUN)

    if synsets:
        synset = synsets[0]
        hypernym_paths = synset.hypernym_paths()
        if hypernym_paths:
            hypernym_path = hypernym_paths[0]
            hypernym_path.reverse()
            return hypernym_path
        else:
            return None
    else:
        return None


def obtain_hypernym_paths(word):
    word_root = wordnet.morphy(word)
    if word_root:
        synsets = wordnet.synsets(word_root, pos=wordnet.NOUN)
    else:
        synsets = wordnet.synsets(word, pos=wordnet.NOUN)

    if synsets:
        synset = synsets[0]
        hypernym_paths = synset.hypernym_paths()
        if hypernym_paths:
            return hypernym_paths
        else:
            return None
    else:
        return None


def sort_dict_by_value(unsorted_dict):
    sorted_tuples = sorted(unsorted_dict.items(), key=lambda x: x[1], reverse=True)

    sorted_dict = {}
    for sorted_tuple in sorted_tuples:
        sorted_dict[sorted_tuple[0]] = sorted_tuple[1]

    return sorted_dict


if __name__ == '__main__':
    print(obtain_hypernym_path("bottle"))

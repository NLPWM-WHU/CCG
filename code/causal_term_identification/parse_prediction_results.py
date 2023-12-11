from file_io import *


def main():
    intervened_file = "ace2005/substitution.tsv"
    prediction_file = "ace2005/substitution_prediction.tsv"
    preliminary_trigger_words_file = "ace2005/substitution100_preliminary_trigger_words.json"

    intervened_instances = read_tsv_file(intervened_file)
    prediction_results = read_tsv_file(prediction_file)
    assert len(intervened_instances) == len(prediction_results)

    preliminary_trigger_words = {}
    for index in range(len(intervened_instances)):
        original_relation = intervened_instances[index][0]
        prediction_relation = prediction_results[index][0]
        if original_relation != prediction_relation:
            instance_id = intervened_instances[index][2]
            trigger_word = intervened_instances[index][3]
            if instance_id not in preliminary_trigger_words.keys():
                preliminary_trigger_words[instance_id] = [trigger_word]
            else:
                if trigger_word not in preliminary_trigger_words[instance_id]:
                    preliminary_trigger_words[instance_id].append(trigger_word)

    for instance_id in preliminary_trigger_words.keys():
        preliminary_trigger_words[instance_id].sort()

    write_json_file(preliminary_trigger_words_file, preliminary_trigger_words)
    print(len(preliminary_trigger_words.keys()))


def main_():
    test_file = "./R-BERT/data/inference_coco.tsv"
    pred_file = "./R-BERT/data/inference_all_prediction.tsv"
    out_file = "./R-BERT/data/inference_bert_filtered.tsv"

    test_instances = read_tsv_file(test_file)
    pred_results = read_tsv_file(pred_file)
    assert len(test_instances) == len(pred_results)

    filtered_instances = []
    # already_ids = []
    for index in range(len(test_instances)):
        if test_instances[index][0] == pred_results[index][0]:
            # if test_instances[index][-1] not in already_ids:
            filtered_instances.append(test_instances[index])
            # already_ids.append(test_instances[index][-1])

    write_tsv_file(out_file, filtered_instances)


if __name__ == '__main__':
    main_()

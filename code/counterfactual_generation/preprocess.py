import os
import csv
import torch
from transformers import AutoTokenizer


class InputExample(object):
    def __init__(self, guid, text, label=None):
        self.guid = guid
        self.text = text
        self.label = label


class InputFeatures(object):
    def __init__(self, input_ids, label_id=None):
        self.input_ids = input_ids
        self.label_id = label_id


class DataProcessor(object):
    def get_train_examples(self, data_dir):
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        raise NotImplementedError()

    def get_labels(self):
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines

    @classmethod
    def _read_csv(cls, input_file, quotechar=None):
        """Reads a comma separated value file."""
        with open(input_file, "r") as f:
            reader = csv.reader(f, quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines


class Processor(DataProcessor):
    def get_train_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return None

    @staticmethod
    def _create_examples(lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text = line[1]
            examples.append(InputExample(guid=guid, text=text))
        return examples


def convert_single_example(ex_index, example, max_length, tokenizer):
    tokens = tokenizer.tokenize(example.text)

    while len(tokens) < max_length:
        tokens.append(tokenizer.eos_token)
    while len(tokens) > max_length:
        tokens.pop()

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    assert len(input_ids) == max_length

    if ex_index < 5:
        print("*** Example ***")
        print("guid: %s" % example.guid)
        print("tokens: %s" % " ".join([str(x) for x in tokens]))
        print("input_ids: %s" % " ".join([str(x) for x in input_ids]))

    feature = InputFeatures(input_ids=input_ids)
    return feature


def convert_examples_to_features(examples, max_length, tokenizer):
    features = []
    for (ex_index, example) in enumerate(examples):
        feature = convert_single_example(ex_index, example, max_length, tokenizer)
        features.append(feature)
    return features


class DatasetLoader(object):
    def __init__(self, args):
        self.data_dir = args.data_dir
        self.processor = Processor()
        self.max_length = args.max_length
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_dir)

    @staticmethod
    def to_tensor(features):
        input_ids = []
        for feature in features:
            input_ids.append(feature.input_ids)
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        return input_ids

    def get_trainset(self):
        train_examples = self.processor.get_train_examples(self.data_dir)
        train_features = convert_examples_to_features(train_examples, self.max_length, self.tokenizer)
        return self.to_tensor(train_features)

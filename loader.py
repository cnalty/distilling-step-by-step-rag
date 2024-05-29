import json
import numpy as np
from datasets import Dataset, DatasetDict

DATASET_ROOT = "datasets"


class DatasetLoader(object):
    def __init__(
        self,
        dataset_name,
        split_map,
    ):
        self.data_root = DATASET_ROOT
        self.dataset_name = dataset_name
        self.split_map = split_map

        assert self.split_map is not None

    def load_from_json(
        self,
    ):
        with open(
            f"{self.data_root}/{self.dataset_name}/{self.dataset_name}_llama3.json"
        ) as f:
            data = json.load(f)

        idxs = np.random.RandomState(seed=0).permutation(len(data))
        train_idxs = idxs[: int(0.8 * len(data))]
        test_idxs = idxs[int(0.8 * len(data)) : int(0.9 * len(data))]
        valid_idxs = idxs[int(0.9 * len(data)) :]

        train_data = [data[i] for i in train_idxs]
        test_data = [data[i] for i in test_idxs]
        valid_data = [data[i] for i in valid_idxs]

        train_dataset = Dataset.from_list(train_data)
        test_dataset = Dataset.from_list(test_data)
        valid_dataset = Dataset.from_list(valid_data)

        datasets = DatasetDict(
            {"train": train_dataset, "test": test_dataset, "valid": valid_dataset}
        )

        return datasets, train_idxs, test_idxs, valid_idxs

    def to_json(self, datasets):
        for k, v in self.split_map.items():
            datasets[v].to_json(
                f"{self.data_root}/{self.dataset_name}/{self.dataset_name}_{k}.json"
            )

    def load_llm_preds(self, split):
        raise NotImplementedError

    def _parse_llm_output(self, output):
        raise NotImplementedError


class CQADatasetLoader(DatasetLoader):
    def __init__(
        self,
    ):
        dataset_name = "cqa"
        split_map = {"train": "train", "test": "test", "valid": "valid"}

        super().__init__(
            dataset_name,
            split_map,
        )

    def load_from_source(self):
        pass

    def load_llm_preds(self, split_idxs):
        labels = list()
        rationales = list()
        with open(f"{self.data_root}/{self.dataset_name}/cqa_llama3.json") as f:
            outputs = json.load(f)

        split_outputs = [outputs[i] for i in split_idxs]

        count_errs = 0
        for i, output in enumerate(split_outputs):
            try:
                rationale, label = self._parse_llm_output(output)
                rationales.append(rationale)
                labels.append(label)
            except Exception as e:
                count_errs += 1
                # print(f"Error parsing LLM output: {i}: {e}")
                pass

        print(f"\nNumber of errors: {count_errs}")
        print(f"Number of processed: {i - count_errs}\n")

        return rationales, labels

    def _post_process(self, datasets):
        return datasets

    def _parse_llm_output(self, output):
        gen_label_text = output["gen_label"]

        # Extract rationale and label
        if " Hence the answer is " in gen_label_text:
            rationale = gen_label_text.split(" Hence the answer is ")[0].strip()
            label = (
                gen_label_text.split(" Hence the answer is ")[1].strip()
            )
        elif " hence the answer is " in gen_label_text:
            rationale = gen_label_text.split(" hence the answer is ")[0].strip()
            label = (
                gen_label_text.split(" hence the answer is ")[1].strip()
            )
        else:
            rationale = gen_label_text
            label = output["org_label"]

        label = label.replace("Answer: ", "").replace(".", "").strip()

        return rationale, label

    def _parse_gpt_output(self, output):
        pass


# ##################
# # Example Usage
# ##################
# dataset_loader = CQADatasetLoader()
# datasets, train_idxs, test_idxs, valid_idxs = dataset_loader.load_from_json()

# valid_llm_rationales, valid_llm_labels = dataset_loader.load_llm_preds(valid_idxs)
# datasets["valid"] = datasets["valid"].add_column("llm_label", valid_llm_labels)
# datasets["valid"] = datasets["valid"].add_column("llm_rationale", valid_llm_rationales)

# train_llm_rationales, train_llm_labels = dataset_loader.load_llm_preds(train_idxs)
# datasets["train"] = datasets["train"].add_column("llm_label", train_llm_labels)
# datasets["train"] = datasets["train"].add_column("llm_rationale", train_llm_rationales)

# test_llm_rationales, test_llm_labels = dataset_loader.load_llm_preds(test_idxs)
# datasets["test"] = datasets["test"].add_column("llm_label", test_llm_labels)
# datasets["test"] = datasets["test"].add_column("llm_rationale", test_llm_rationales)

# dataset_loader.to_json(datasets)

import os
from typing import Any, List, Union, Dict
import torch
import pandas as pd
from trl import DataCollatorForCompletionOnlyLM
from datasets import Dataset, interleave_datasets, load_dataset, load_from_disk
import model_utils


class MultiDatasetDataCollatorCompletion(DataCollatorForCompletionOnlyLM):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        is_dpo = examples[0].get("logps", None)
        keys = ["input_ids", "attention_mask"]
        if is_dpo is not None:
            keys.append("logps")

        clean_examples = [{key: ex[key] for key in keys} for ex in examples]
        safe_examples = [
            {key: ex["Safe_Model"][key] for key in keys}
            for ex in examples
            if isinstance(ex["Safe_Model"], dict)
        ]
        clean_examples += safe_examples
        batch = super().torch_call(clean_examples)
        batch["dataset_id"] = torch.concat(
            [
                torch.tensor([ex["dataset_id"] for ex in examples]),
                torch.ones(len(safe_examples)),
            ]  # Hardcoded value of 1 for safe examples
        )

        return batch


def get_dataset_ids():
    return [0, 1, 2]


def get_dataset_id(key):
    dataset_ids = {"away": 0, "toward": 1, "utility": 2}
    return dataset_ids[key]


def get_prompt_formatting_func_and_collator(model_name, tokenizer, collator_type="multi"):
    first_user_msg, user_chat_template, response_template, response_key = model_utils.get_chat_template(
        model_name
    )

    dataset_text_field, dataset_target_field = get_dataset_text_and_target_field()

    def prompt_formatting_func(sample, input_only=False):
        all_formatted_prompts = []
        for i in range(len(sample[dataset_text_field])):
            formatted_prompt = ""

            formatted_instruction = first_user_msg.format(instruction=sample[dataset_text_field][i][0])
            if input_only:
                formatted_prompt = formatted_instruction + response_key
            else:
                formatted_target = response_template.format(target=sample[dataset_target_field][i][0])
                formatted_prompt += formatted_instruction + formatted_target
                for instruction, target in zip(
                    sample[dataset_text_field][i][1:], sample[dataset_target_field][i][1:]
                ):
                    formatted_instruction = user_chat_template.format(instruction=instruction)
                    formatted_target = response_template.format(target=target)
                    formatted_prompt += formatted_instruction + formatted_target

            all_formatted_prompts.append(formatted_prompt)

        return all_formatted_prompts

    # NOTE the phi-3 tokenizer adds the SPIECE_UNDERLINE token (29871) to an encoded <|assistant|> token if not other token is present except for \n, which messes with the matching
    if model_name == "phi-3":
        response_key = [13, 32001, 13]

    if collator_type == "multi":
        collator = MultiDatasetDataCollatorCompletion(response_key, tokenizer=tokenizer)
    elif collator_type == "single":
        collator = DataCollatorForCompletionOnlyLM(response_key, tokenizer=tokenizer)
    else:
        raise ValueError(f"Collator type {collator_type} not supported")

    return prompt_formatting_func, collator, response_key


DATASETS = [
    "ultrachat",
    "ultrachat_200k",
    "adv_training_behaviors",
    "adv_val_behaviors",
    "adv_test_behaviors",
    "adv_training_safe_prompts",
]


def load_specific_dataset(data_path, dataset_name, split=None, multiple_targets=False):
    if dataset_name not in DATASETS:
        raise NotImplementedError(f"Dataset {dataset_name} not supported, choose from: {DATASETS}")

    if dataset_name == "ultrachat":
        dataset_path = data_path + "utility/ultrachat"
        if os.path.isdir(dataset_path) and len(os.listdir(dataset_path)) > 0:
            dataset = load_from_disk(dataset_path)
        else:
            dataset = load_dataset("stingning/ultrachat", split="train")
            dataset.save_to_disk(dataset_path)
        dataset = dataset.map(
            lambda example: {
                "User": example["data"][::2],
                "Model": [ans for ans in example["data"][1::2]],
            }
        )
        frac = int(0.9 * len(dataset))
        dataset = dataset.select(range(frac)), dataset.select(range(frac, len(dataset)))
    if dataset_name == "ultrachat_200k":
        dataset_path_train = data_path + "utility/ultrachat_200k_train"
        if os.path.isdir(dataset_path_train) and len(os.listdir(dataset_path_train)) > 0:
            dataset_train = load_from_disk(dataset_path_train)
        else:
            dataset_train = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
            dataset_train.save_to_disk(dataset_path_train)

        dataset_train = dataset_train.map(
            lambda example: {
                "User": [ans["content"] for ans in example["messages"][::2]],
                "Model": [ans["content"] for ans in example["messages"][1::2]],
            }
        )
        # Assuming each token is on average ~4 chars, limit to ~256tokens
        dataset_train = dataset_train.filter(lambda example: len(example["User"][0]) < 768)

        dataset_path_eval = data_path + "utility/ultrachat_200k_eval"
        if os.path.isdir(dataset_path_eval) and len(os.listdir(dataset_path_eval)) > 0:
            dataset_eval = load_from_disk(dataset_path_eval)
        else:
            dataset_eval = load_dataset("HuggingFaceH4/ultrachat_200k", split="test_sft")
            dataset_eval.save_to_disk(dataset_path_eval)
        dataset_eval = dataset_eval.map(
            lambda example: {
                "User": [ans["content"] for ans in example["messages"][::2]],
                "Model": [ans["content"] for ans in example["messages"][1::2]],
            }
        )
        # Assuming each token is on average ~4 chars, limit to ~256tokens
        dataset_eval = dataset_eval.filter(lambda example: len(example["User"][0]) < 768)
        dataset = dataset_train, dataset_eval
    elif dataset_name == "adv_training_behaviors":
        train_behavior_filename = (
            data_path + "behavior_datasets/extra_behavior_datasets/adv_training_behaviors.csv"
        )
        train_target_filename = data_path + "optimizer_targets/extra_targets/adv_training_targets.json"
        df = create_df_from_behavior_and_target(train_behavior_filename, train_target_filename)
        df = df.rename(columns={"Behavior": "User", "Target": "Model", "Safe_Response": "Safe_Model"})
        dataset = Dataset.from_pandas(df)

        if not multiple_targets:
            dataset = dataset.map(
                lambda batch: {
                    "User": [[batch["User"][0]] for _ in range(len(batch["Model"][0]))],
                    "Model": [[target] for target in batch["Model"][0]],
                },
                batched=True,
                batch_size=1,
            )
        else:
            dataset = dataset.map(
                lambda batch: {
                    "User": [batch["User"]],
                    "Model": batch["Model"],
                },
                batched=True,
                batch_size=1,
            )

    elif dataset_name == "adv_val_behaviors":
        val_behavior_filename = data_path + "behavior_datasets/harmbench_behaviors_text_val.csv"
        val_target_filename = data_path + "optimizer_targets/extra_targets/adv_training_val_targets.json"
        df = create_df_from_behavior_and_target(val_behavior_filename, val_target_filename)
        df = df.rename(columns={"Behavior": "User", "Target": "Model", "Safe_Response": "Safe_Model"})
        dataset = Dataset.from_pandas(df)
        dataset = dataset.map(
            lambda batch: {
                "User": [batch["User"]],
                "Model": [[batch["Model"][0][0]]],
            },
            batched=True,
            batch_size=1,
        )
    elif dataset_name == "adv_test_behaviors":
        test_behavior_filename = data_path + "behavior_datasets/harmbench_behaviors_text_all.csv"
        test_target_filename = data_path + "optimizer_targets/harmbench_targets_text.json"
        df = create_df_from_behavior_and_target(test_behavior_filename, test_target_filename)
        df = df.rename(columns={"Behavior": "User", "Target": "Model", "Safe_Response": "Safe_Model"})
        dataset = Dataset.from_pandas(df)
        dataset = dataset.map(
            lambda batch: {
                "User": [batch["User"]],
                "Model": [[batch["Model"][0]]],
            },
            batched=True,
            batch_size=1,
        )
    elif dataset_name == "adv_training_safe_prompts":
        safe_prompts_file_name = data_path + "safe_responses/adv_training_behaviors_safe_responses.csv"
        df = pd.read_csv(safe_prompts_file_name)
        df = df.rename(columns={"Behavior": "User", "Target": "Model", "Safe_Response": "Safe_Model"})
        dataset = Dataset.from_pandas(df)

    return dataset


def create_df_from_behavior_and_target(behavior_path, target_path):
    df_behavior = pd.read_csv(
        behavior_path,
        usecols=["Behavior", "BehaviorID"],
        encoding="utf-8",
        encoding_errors="replace",
    )
    df_target = pd.read_json(target_path, typ="series").reset_index()
    df_target.columns = ["BehaviorID", "Target"]
    df = df_behavior.merge(df_target, on="BehaviorID")
    df = df[["Behavior", "Target"]]
    return df


def load_adversarial_training_data(
    data_path,
    utility_data,
    probabilities,
    model_name,
    tokenizer,
    stopping_strategy="first_exhausted",
    diverse_safe_answers=False,
    restricted_trainingset_size=None,
):
    train_behavior_filename = (
        data_path + "behavior_datasets/extra_behavior_datasets/adv_training_behaviors.csv"
    )
    train_target_filename = data_path + "optimizer_targets/extra_targets/adv_training_targets.json"

    val_behavior_filename = data_path + "behavior_datasets/harmbench_behaviors_text_val.csv"
    val_target_filename = data_path + "optimizer_targets/extra_targets/adv_training_val_targets.json"

    df_train = create_df_from_behavior_and_target(train_behavior_filename, train_target_filename)
    if diverse_safe_answers:
        safe_data_path = data_path + "safe_responses/adv_training_behaviors_safe_responses_longer.csv"
    else:
        safe_data_path = data_path + "safe_responses/adv_training_behaviors_safe_responses.csv"

    df_train_safe_prompts = pd.read_csv(safe_data_path)
    df_train = pd.merge(df_train, df_train_safe_prompts, on="Behavior")
    df_train = df_train.rename(columns={"Behavior": "User", "Target": "Model", "Safe_Response": "Safe_Model"})
    df_val = create_df_from_behavior_and_target(val_behavior_filename, val_target_filename)
    df_val = df_val.rename(columns={"Behavior": "User", "Target": "Model"})
    df_val["Safe_Model"] = "Sorry, I cannot do that."

    train_data = Dataset.from_pandas(df_train)
    eval_data = Dataset.from_pandas(df_val)
    # TODO change eval safe response

    # add dataset ids and duplicate behaviours with different targets
    ## train
    train_data = train_data.map(
        lambda batch: {
            "dataset_id": [0 for _ in range(len(batch["Model"][0]))],
            "User": [[batch["User"][0]] for _ in range(len(batch["Model"][0]))],
            "Model": [[target] for target in batch["Model"][0]],
            "Safe_Model": [[batch["Safe_Model"][0]] for _ in range(len(batch["Model"][0]))],
        },
        batched=True,
        batch_size=1,
    )
    ## eval
    eval_data = eval_data.map(
        lambda batch: {
            "dataset_id": [0 for _ in range(len(batch["Model"][0]))],
            "User": [[batch["User"][0]] for _ in range(len(batch["Model"][0]))],
            "Model": [[target] for target in batch["Model"][0]],
            "Safe_Model": [[batch["Safe_Model"][0]] for _ in range(len(batch["Model"][0]))],
        },
        batched=True,
        batch_size=1,
    )

    # format and tokenize safe response
    first_user_msg, user_chat_template, response_template, _ = model_utils.get_chat_template(model_name)
    ## train
    train_data = train_data.map(
        lambda example: {
            "Safe_Model": tokenizer(
                first_user_msg.format(instruction=example["User"][0])
                + response_template.format(target=example["Safe_Model"][0])
            ),
        }
    )
    ## eval
    eval_data = eval_data.map(
        lambda example: {
            "Safe_Model": tokenizer(
                first_user_msg.format(instruction=example["User"][0])
                + response_template.format(target=example["Safe_Model"][0])
            ),
        }
    )

    # add utility data
    if utility_data is not None and utility_data != "None":
        utility_train_data, utility_eval_data = load_specific_dataset(data_path, utility_data)
        #### Format train data ####
        utility_train_data = utility_train_data.map(
            lambda example: {"dataset_id": 2, "Safe_Model": None, **example}, num_proc=16
        )
        columns = [k for k in utility_train_data.column_names if k not in train_data.column_names]
        utility_train_data = utility_train_data.remove_columns(columns)
        #### Format eval data ####
        utility_eval_data = utility_eval_data.map(
            lambda example: {"dataset_id": 2, "Safe_Model": None, **example}, num_proc=16
        )
        columns = [k for k in utility_eval_data.column_names if k not in eval_data.column_names]
        utility_eval_data = utility_eval_data.remove_columns(columns)
        # TODO may want to reload the dataloader at each epoch to get different samples from larger dataset
        train_data = interleave_datasets(
            [train_data, utility_train_data],
            probabilities=probabilities,
            stopping_strategy=stopping_strategy,
        )
        eval_data = interleave_datasets(
            [eval_data, utility_eval_data],
            probabilities=probabilities,
            stopping_strategy=stopping_strategy,
        )
        eval_data = eval_data.select(range(256))

    if restricted_trainingset_size is not None:
        train_data = train_data.select(range(restricted_trainingset_size))

    return train_data, eval_data


def get_dataset_text_and_target_field():
    dataset_text_field = "User"
    dataset_target_field = "Model"
    return dataset_text_field, dataset_target_field

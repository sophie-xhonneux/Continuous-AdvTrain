from omegaconf import OmegaConf
from typing import List, Optional, Any
from dataclasses import dataclass, field
import hydra
from hydra.core.config_store import ConfigStore
import os
import fcntl
import logging
from hydra.types import RunMode

import adversarial_training
import database_handling
import model_utils
import eval_config


@dataclass
class MyLoraConfig:
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    r: int = 64
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    target_modules: Any = "all-linear"


@dataclass
class MyBnBConfig:
    load_in_4bit: bool = True
    bnb_4bit_use_double_quant: bool = False
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "bfloat16"


@dataclass
class MyTrainingConfig:
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 8
    gradient_accumulation_steps: int = 1
    optim: str = "paged_adamw_32bit"
    save_steps: int = 50
    save_total_limit: int = 1
    save_strategy: str = "steps"
    logging_steps: int = 25
    learning_rate: float = 2e-4
    weight_decay: float = 0.001
    fp16: bool = True
    bf16: bool = False
    max_grad_norm: float = 0.3
    max_steps: int = -1
    warmup_ratio = 0.03
    group_by_length: bool = False
    lr_scheduler_type: str = "cosine"
    do_eval: bool = False
    evaluation_strategy: str = "no"
    eval_steps: Optional[float] = None
    remove_unused_columns: bool = False


@dataclass
class AdversarialTrainingConfig:
    iters: int = 100
    opt_config: dict = field(default_factory=lambda: {"type": "rms", "lr": 1e-4})
    eps: float = 0.2
    debug: int = 3
    init_type: str = "instruction"
    attack_type: str = "embedding"


@dataclass
class TrainerHyperParams:
    away_weight: float = 1
    toward_weight: float = 1
    utility_weight: float = 1
    ema_weight: float = 0
    padding_side: str = "left"
    away_cutoff: float = -5
    toward_cutoff: float = 0.0
    away_loss_type: str = "negative_cross_entropy"
    restart_count: int = 0
    trainer_type: str = "ul"
    dtype: str = "bf16"
    # DPO
    dpo_loss_type: str = "ipo"
    dpo_beta: float = 0.1
    dpo_weight: float = 1
    do_online_dpo: bool = False
    dpo_label_smoothing: float = 0.25


@dataclass
class DatasetConfig:
    data_path: Optional[str] = "../../../data/"
    dataset_name: Optional[str] = "adv_training_behaviors"
    utility_data: Optional[str] = "ultrachat"  # TODO remove this and use dataset_name
    probabilities: List[float] = field(default_factory=lambda: [0.5, 0.5])
    stopping_strategy: str = "first_exhausted"
    restricted_trainingset_size: Optional[int] = None
    diverse_safe_answers: bool = False
    subset_ids: List = field(default_factory=lambda: [])


@dataclass
class MySFTTrainerConfig:
    packing: bool = False
    max_seq_length: int = 256  # TODO more or less needed?


@dataclass
class PathConfig:
    experiments_path: str = "./experiments"
    logging_path: str = "./results"
    model_path: str = "google/gemma-2b-it"
    model_name: Optional[str] = None
    checkpoint_path: Optional[str] = ""
    load_checkpoint_path: Optional[str] = None
    load_checkpoint: bool = False


@dataclass
class GlobalConfig:
    experiment: str = "NOT SET"
    experiment_id: Optional[int] = None
    skip_existing_experiment: bool = False
    finished_experiment: bool = False
    model_name: Optional[str] = None
    debug: bool = False
    peft: Optional[MyLoraConfig] = field(default_factory=MyLoraConfig)
    bnb: Optional[MyBnBConfig] = field(default_factory=MyBnBConfig)
    training: MyTrainingConfig = field(default_factory=MyTrainingConfig)
    adversarial: AdversarialTrainingConfig = field(default_factory=AdversarialTrainingConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    eval_dataset: eval_config.EvalDatasetConfig = field(default_factory=eval_config.EvalDatasetConfig)
    sfttrainer: MySFTTrainerConfig = field(default_factory=MySFTTrainerConfig)
    trainer_hparams: TrainerHyperParams = field(default_factory=TrainerHyperParams)
    path: PathConfig = field(default_factory=PathConfig)


cs = ConfigStore.instance()
cs.store(group="path", name="base_path", node=PathConfig)
cs.store(group="adversarial", name="base_adversarial", node=AdversarialTrainingConfig)
cs.store(group="dataset", name="base_dataset", node=DatasetConfig)
cs.store(group="eval_dataset", name="base_eval_dataset", node=eval_config.EvalDatasetConfig)
cs.store(group="training", name="base_training", node=MyTrainingConfig)
cs.store(group="peft", name="base_peft", node=MyLoraConfig)
cs.store(group="bnb", name="base_bnb", node=MyBnBConfig)
cs.store(group="path", name="base_path", node=PathConfig)
cs.store(group="trainer_hparams", name="base_trainer_hparams", node=TrainerHyperParams)
cs.store(group="sfttrainer", name="base_sft_trainer", node=MySFTTrainerConfig)
cs.store(name="base_config", node=GlobalConfig)

EXPERIMENTS = [
    "adversarial_training",
]


def acquireLock(directory):
    """acquire exclusive lock file access"""
    locked_file_descriptor = open(directory + "lockfile.LOCK", "w+")
    fcntl.lockf(locked_file_descriptor, fcntl.LOCK_EX)
    return locked_file_descriptor


def releaseLock(locked_file_descriptor):
    """release exclusive lock file access"""
    locked_file_descriptor.close()


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: GlobalConfig) -> None:
    print("Starting program")

    if cfg.experiment not in EXPERIMENTS:
        raise ValueError(f"Invalid experiment: {cfg.experiment}. Choose from: {EXPERIMENTS}")

    if cfg.path.model_name is None:
        cfg.model_name = model_utils.get_model_name(cfg.path.model_path)
    else:
        cfg.model_name = cfg.path.model_name

    hydra_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    cfg.path.logging_path = hydra_dir
    if hydra.core.hydra_config.HydraConfig.get().mode == RunMode.MULTIRUN:
        cfg.path.checkpoint_path = cfg.path.checkpoint_path + "/" + hydra_dir.strip().split("/")[-1]
    else:
        cfg.path.checkpoint_path = cfg.path.logging_path

    logging.info(f"Running experiment: {cfg.experiment}")
    logging.info(f"Results will be saved to {cfg.path.logging_path}")
    logging.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    db_path = cfg.path.experiments_path
    if not os.path.exists(db_path):
        os.makedirs(db_path)
    lock = acquireLock(db_path)
    if not cfg.debug:
        experiment_path = database_handling.init_experiment(cfg)
    releaseLock(lock)

    if cfg.experiment == "adversarial_training":
        adversarial_training.adversarial_training_loop(
            cfg.model_name,
            dict(cfg.path),
            dict(cfg.adversarial),
            dict(cfg.dataset),
            dict(cfg.training),
            dict(cfg.peft) if cfg.peft is not None else None,
            dict(cfg.bnb) if cfg.bnb is not None else None,
            dict(cfg.sfttrainer),
            dict(cfg.trainer_hparams),
        )

    lock = acquireLock(db_path)
    if not cfg.debug:
        database_handling.update_experiment_file(experiment_path, "finished_experiment", True)
    releaseLock(lock)
    print("Experiment finished")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.info(e)
        raise e


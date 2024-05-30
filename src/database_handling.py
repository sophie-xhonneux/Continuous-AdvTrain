from omegaconf import OmegaConf
import tinydb
import pandas as pd
import json
import hydra
import os


def update_experiment_file(experiment_path, key, val):
    with open(experiment_path, "r") as f:
        cfg = json.load(f)

    cfg[key] = val

    with open(experiment_path, "w") as f:
        json.dump(cfg, f)


def flatten_dict(d, parent_key="", sep="_"):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def init_experiment(cfg):
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    cfg_dict = OmegaConf.to_container(cfg)
    cfg_dict["finished_experiment"] = False

    experiment_name = str(cfg.experiment_id)
    if hydra.core.hydra_config.HydraConfig.get().mode == hydra.types.RunMode.MULTIRUN:
        experiment_name = experiment_name + "-" + cfg.path.logging_path.split("/")[-1]
    experiment_path = cfg.path.experiments_path + experiment_name + ".json"

    cfg_dict["experiment_path"] = experiment_path

    with open(experiment_path, "w") as f:
        json.dump(cfg_dict, f)

    return experiment_path


def create_db(experiments_path):
    db = tinydb.TinyDB(experiments_path + "db_experiments.json")
    experiment_files = os.listdir(experiments_path)
    valid_experiments = []
    for exp_file in experiment_files:
        if (
            exp_file.endswith(".json")
            and exp_file != "db_experiments.json"
            and exp_file != "experiments.json"
        ):
            with open(experiments_path + exp_file, "r") as f:
                try:
                    cfg = json.load(f)
                except:
                    print("renaming")
                    os.rename(experiments_path + exp_file, experiments_path + exp_file + ".old")
                    text = f.read()
                    with open(experiments_path + exp_file, "w") as f_new:
                        f_new.write(text)
                    cfg = json.load(f_new)

            try:
                existing_experiments = len(
                    db.search(tinydb.Query().path.logging_path == cfg["path"]["logging_path"])
                )
            except Exception as e:
                print("Json error with experiment:", exp_file)
                raise e
            cfg["filename"] = exp_file
            if existing_experiments == 0:
                db.insert(cfg)
            elif existing_experiments == 1:
                db.remove(tinydb.Query().path.logging_path == cfg["path"]["logging_path"])
                try:
                    db.insert(cfg)
                except:
                    raise ValueError("Could not insert experiment:", exp_file)
            valid_experiments.append(cfg["path"]["logging_path"])
    for exp in db:
        if exp["path"]["logging_path"] not in valid_experiments:
            db.remove(tinydb.Query().path.logging_path == exp["path"]["logging_path"])
    return db


def db_update_experiment_with_id(db, cfg, key, val):
    q = tinydb.Query()
    db.update({key: val}, q.experiment_id == cfg.experiment_id)


def db_search_experiment_with_id(cfg):
    db_path = cfg.path.experiments_path
    db = tinydb.TinyDB(db_path + "experiments.json")
    q = tinydb.Query()

    exp = db.search(q.experiment_id == cfg.experiment_id)

    if len(exp) == 0:
        raise ValueError("Experiment not found")
    if len(exp) > 1:
        raise ValueError("Multiple experiments found with same id")

    return exp[0]


def db_entry_exists(db, cfg):
    def remove_nonrelevant(cfg_dict):
        cfg_dict.pop("experiment_id", None)
        cfg_dict.pop("skip_existing_experiment", None)
        cfg_dict.pop("finished_experiment", None)
        cfg_dict.pop("debug", None)
        cfg_dict.pop("model_name", None)
        cfg_dict["path"].pop("logging_path", None)
        cfg_dict["path"].pop("experiments_path", None)
        cfg_dict["path"].pop("checkpoint_path", None)
        cfg_dict["dataset"].pop("data_path", None)

    search_dict = OmegaConf.to_container(cfg, resolve=True)
    remove_nonrelevant(search_dict)

    db_entries = db_search_with_dict(db, search_dict)

    return len(db_entries) > 0


def db_search_with_dict(db, search_dict):
    def predicate(obj, requirements):
        def match(item, criteria):
            if not isinstance(criteria, dict) and not isinstance(criteria, list):
                return item == criteria
            for k, v in criteria.items():
                if k not in item:
                    return True
                if isinstance(v, dict):
                    if not isinstance(item[k], dict) or not match(item[k], v):
                        return False
                elif isinstance(v, list):
                    if not isinstance(item[k], list) or not all(
                        match(sub_item, v[0]) for sub_item in item[k]
                    ):
                        return False
                else:
                    if item[k] != v:
                        return False
            return True

        return match(obj, requirements)

    run = db.search(lambda obj: predicate(obj, search_dict))
    return run


def db_get_training_and_corresponding_eval_experiments(db):
    df = pd.DataFrame()
    search_dict = {"experiment": "adversarial_training"}
    train_runs = db_search_with_dict(db, search_dict)
    for tr in train_runs:
        df_row = flatten_dict(tr)
        test_runs = db_get_eval_experiments_from_train(db, tr)

        for test_r in test_runs:
            df_row.update({"experiment_" + test_r["experiment"]: test_r["finished_experiment"]})
        df_row = pd.DataFrame([df_row])
        df = pd.concat([df, df_row], ignore_index=True)

    return df.fillna(False)


def db_get_eval_experiments_from_train(db, train_exp):
    load_checkpoint_path = train_exp["path"]["checkpoint_path"] + "/final_model"
    q = tinydb.Query()
    exp = db.search(q.path.load_checkpoint_path == load_checkpoint_path)
    return exp


def db_get_train_experiment_from_eval(db, eval_exp):
    load_checkpoint_path = eval_exp["path"]["load_checkpoint_path"]
    if load_checkpoint_path is None or load_checkpoint_path == "None":
        return None
    else:
        q = tinydb.Query()
        exp = db.search(q.path.checkpoint_path == load_checkpoint_path.replace("/final_model", ""))
    if len(exp) > 1:
        print("Multiple training experiments found for eval experiment", exp)
    if len(exp) == 0:
        q = tinydb.Query()
        exp = db.search(q.path.checkpoint_path == load_checkpoint_path.replace("/final_model", ""))
        print("did not find training run for exp", eval_exp["path"]["logging_path"])
        print("checkpoint", load_checkpoint_path)
        print("exp", eval_exp["filename"])
        return None
    return exp[0]

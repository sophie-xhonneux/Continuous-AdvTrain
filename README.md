# Continuous-AdvTrain

This is a repository for the adversarial training code of our paper https://arxiv.org/abs/2405.15589 .

## Installation

1. Clone this repository with `git clone git@github.com:sophie-xhonneux/Continuous-AdvTrain.git`
2. Install the requirements `pip install -r requirements.txt`

## Running the code

1. Create a config in `config/path` (see `example_path.yaml`)
2. Run the code with `python src/run_experiments.py --config-name=adv_train_ul path=example_path`

You can also run the IPO experiments by replacing `adv_train_ul` with `adv_train_ipo`. Moreover, hydra allows you to override any hyperparameters from the commandline (e.g. add `adversarial.eps=0.075`) or you can create a new config file under the `config` folder. See the paper for the exact hyperparameters.

## Data

The data is in the data folder is from the Harmbench repository (`https://github.com/centerforaisafety/HarmBench`) with the exception of a couple files we created as part of this paper.

## Citation

If you used this code, please cite our paper:

```
@misc{xhonneux2024efficient,
      title={Efficient Adversarial Training in LLMs with Continuous Attacks}, 
      author={Sophie Xhonneux and Alessandro Sordoni and Stephan Günnemann and Gauthier Gidel and Leo Schwinn},
      year={2024},
      eprint={2405.15589},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
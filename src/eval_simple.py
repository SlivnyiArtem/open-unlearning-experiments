import hydra
from omegaconf import DictConfig
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM

from data import get_data, get_collators
from model import get_model
from trainer import load_trainer
from evals import get_evaluator
from trainer.utils import seed_everything


def main():
    config = PeftConfig.from_pretrained("../saves/unlearn/opt_13_unlearn_gradDiff")
    base_model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path)
    model = PeftModel.from_pretrained(base_model, "../saves/unlearn/opt_13_unlearn_gradDiff")

    # config = PeftConfig.from_pretrained("../saves/unlearn/opt_13_unlearn_gradDiff")
    # base_model = PeftModel.from_pretrained("../saves/unlearn/opt_13_unlearn_gradDiff")
    # base_model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path)
    print(model)



if __name__ == "__main__":
    main()

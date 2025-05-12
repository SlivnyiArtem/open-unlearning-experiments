from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from omegaconf import DictConfig, open_dict
import os
from peft import AdaLoraConfig, get_peft_model
import torch
import logging

hf_home = os.getenv("HF_HOME", default=None)


logger = logging.getLogger(__name__)


def get_dtype(model_args):
    with open_dict(model_args):
        torch_dtype = model_args.pop("torch_dtype", None)
    if model_args.get("attn_implementation", None) == "flash_attention_2":
        # This check handles https://github.com/Dao-AILab/flash-attention/blob/7153673c1a3c7753c38e4c10ef2c98a02be5f778/flash_attn/flash_attn_triton.py#L820
        # If you want to run at other precisions consider running "training or inference using
        # Automatic Mixed-Precision via the `with torch.autocast(device_type='torch_device'):`
        # decorator" or using an attn_implementation compatible with the precision in the model
        # config.
        assert torch_dtype in ["float16", "bfloat16"], ValueError(
            f"Invalid torch_dtype '{torch_dtype}' for the requested attention "
            f"implementation: 'flash_attention_2'. Supported types are 'float16' "
            f"and 'bfloat16'."
        )
    if torch_dtype == "float16":
        return torch.float16
    elif torch_dtype == "bfloat16":
        return torch.bfloat16
    return torch.float32


def get_model(model_cfg: DictConfig):
    assert model_cfg is not None and model_cfg.model_args is not None, ValueError(
        "Model config not found or model_args absent in configs/model."
    )
    model_args = model_cfg.model_args
    tokenizer_args = model_cfg.tokenizer_args
    torch_dtype = get_dtype(model_args)
    try:
        if model_args.get("quant", False):
            model_args.__delattr__("quant") #Убрать костыль
            print("QuantIsTrue")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch_dtype,
            )
            model = AutoModelForCausalLM.from_pretrained(
                **model_args,
                quantization_config=bnb_config,
                cache_dir=hf_home
            )
            # model.to('cuda')

            ada_lora_config = AdaLoraConfig(
                init_r=8,  # начальный ранг
                target_r=4,  # целевой ранг (может динамически уменьшаться)
                beta1=0.85,  # коэффициент для адаптивного ранга
                beta2=0.85,  # коэффициент для адаптивного ранга
                tinit=20,  # шаги до начала адаптации ранга
                tfinal=50,  # шаги до завершения адаптации ранга
                total_step=200,
                deltaT=10,  # интервал обновления ранга
                lora_alpha=32,  # коэффициент масштабирования
                lora_dropout=0.1,
                target_modules=["q_proj", "v_proj"],  # модули для адаптации
                task_type="CAUSAL_LM",  # тип задачи (для языковых моделей)
            )

            model = get_peft_model(model, ada_lora_config)
        else:
            model = AutoModelForCausalLM.from_pretrained(
                torch_dtype=torch_dtype, **model_args, cache_dir=hf_home
            )
            model.to('cuda')

    except Exception as e:
        logger.warning(
            f"Model {model_args.pretrained_model_name_or_path} requested with {model_cfg.model_args}"
        )
        raise ValueError(
            f"Error {e} while fetching model using AutoModelForCausalLM.from_pretrained()."
        )
    tokenizer = get_tokenizer(tokenizer_args)
    return model, tokenizer


def _add_or_replace_eos_token(tokenizer, eos_token: str) -> None:
    is_added = tokenizer.eos_token_id is None
    num_added_tokens = tokenizer.add_special_tokens({"eos_token": eos_token})

    if is_added:
        logger.info("Add eos token: {}".format(tokenizer.eos_token))
    else:
        logger.info("Replace eos token: {}".format(tokenizer.eos_token))

    if num_added_tokens > 0:
        logger.info("New tokens have been added, make sure `resize_vocab` is True.")


def get_tokenizer(tokenizer_cfg: DictConfig):
    try:
        tokenizer = AutoTokenizer.from_pretrained(**tokenizer_cfg, cache_dir=hf_home)
    except Exception as e:
        error_message = (
            f"{'--' * 40}\n"
            f"Error {e} fetching tokenizer using AutoTokenizer.\n"
            f"Tokenizer requested from path: {tokenizer_cfg.get('pretrained_model_name_or_path', None)}\n"
            f"Full tokenizer config: {tokenizer_cfg}\n"
            f"{'--' * 40}"
        )
        raise RuntimeError(error_message)

    if tokenizer.eos_token_id is None:
        logger.info("replacing eos_token with <|endoftext|>")
        _add_or_replace_eos_token(tokenizer, eos_token="<|endoftext|>")

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Setting pad_token as eos token: {}".format(tokenizer.pad_token))

    return tokenizer

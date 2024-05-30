from transformers import (
    MistralForCausalLM,
    GemmaForCausalLM,
    LlamaForCausalLM,
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaTokenizer,
    LlamaTokenizerFast,
    Phi3ForCausalLM,
)

import torch
from peft import PeftModel


def load_model_and_tokenizer(model_path, bnb_config=None, padding_side="left", dtype="bf16"):
    if dtype == "bf16":
        torch_dtype = torch.bfloat16
    elif dtype == "fp16":
        torch_dtype = torch.float16
    elif dtype == "fp32":
        torch_dtype = torch.float32
    else:
        torch_dtype = dtype
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",  # "cuda:0",
        quantization_config=bnb_config,
        low_cpu_mem_usage=True,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )

    model.config.use_cache = False  # Last attention and keys not needed
    model.config.pretraining_tp = 1  # Fast but inaccurate computation for llama2

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    if isinstance(tokenizer, LlamaTokenizer) or isinstance(tokenizer, LlamaTokenizerFast):
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = padding_side

    return model, tokenizer


def load_checkpoint(
    model,
    path_config,
):
    checkpoint_path = path_config["load_checkpoint_path"]
    model = PeftModel.from_pretrained(model, checkpoint_path, is_trainable=False)
    return model


def get_chat_template(model_name):
    found = 0

    if "gemma" in model_name:
        found += 1
        first_user_msg = "<start_of_turn>user\n{instruction}<end_of_turn>\n"
        user_chat_template = "<start_of_turn>user\n{instruction}<end_of_turn>\n"
        response_key = "<start_of_turn>model\n"
        response_template = response_key + "{target}<end_of_turn>\n"
    elif "llama2" == model_name or "llama-2" == model_name:
        found += 1
        first_user_msg = """[INST] <<SYS>>
You are a helpful, respectful and honest assistant.
<</SYS>>

{instruction} """
        user_chat_template = "<s>[INST] {instruction} "
        response_key = "[/INST]"
        # Llama2 tokenizer does not satisfy enc(a+b) = enc(a) + enc(b)
        # Llama2 expects there to be a space token after [/INST]
        # Since we tokenize the prompt plus response in one go, we need two spaces
        response_template = response_key + "  {target} </s>"
    elif "safe-llama2" == model_name:
        found += 1
        first_user_msg = """[INST] <<SYS>>
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
<</SYS>>

{instruction} """
        user_chat_template = "<s>[INST] {instruction} "
        response_key = "[/INST]"
        response_template = response_key + " {target} </s>"
    elif "mistral-instruct" == model_name:
        found += 1
        first_user_msg = "[INST] {instruction} "
        user_chat_template = "[INST] {instruction} "
        response_key = "[/INST]"
        response_template = response_key + " {target} </s>"
    elif "mistral" == model_name:
        found += 1
        first_user_msg = """<|user|>\n{instruction}</s>"""
        user_chat_template = "\n<|user|>\n{instruction}</s>"
        response_key = "\n<|assistant|>\n"
        response_template = response_key + "{target}</s>"
    elif "phi" in model_name:
        found += 1
        first_user_msg = "<|system|>\nYou are a helpful AI assistant.<|end|>\n<|user|>\n{instruction}<|end|>"
        user_chat_template = "<|user|>\n{instruction}<|end|>"
        response_key = "\n<|assistant|>\n"
        response_template = response_key + "{target}<|end|>\n"

    if found == 0:
        raise NotImplementedError(f"Model {model_name} not supported")
    if found > 1:
        raise ValueError(f"Model {model_name} is ambiguous")

    return first_user_msg, user_chat_template, response_template, response_key


def get_embed_weights(model):
    if type(model) == GemmaForCausalLM:
        return model.model.embed_tokens.weight
    elif type(model) == LlamaForCausalLM:
        return model.model.embed_tokens.weight
    elif type(model) == MistralForCausalLM:
        return model.model.embed_tokens.weight
    elif type(model) == Phi3ForCausalLM:
        return model.model.embed_tokens.weight
    else:
        return model.model.embed_tokens.weight


def get_model_name(model_path):
    model_name = model_path.split("/")[-1].lower()
    if "gemma-1.1-2b-it" in model_name:
        return "gemma-1.1-2b-it"
    elif "gemma-2b-it" in model_name:
        return "gemma-2b-it"
    elif "gemma-1.1-7b-it" in model_name:
        return "gemma-1.1-7b-it"
    elif "gemma-2b" in model_name:
        return "gemma-2b"
    elif "harmbench" in model_name:  # NOTE also has llama-2 in the name so order is important
        return "harmbench"
    elif "llama-2" in model_name:
        return "llama-2"
    elif "phi-3" in model_name:
        return "phi-3"
    elif "mistral" in model_name:
        return "mistral"
    elif "r2d2" in model_name:
        return "mistral"
    else:
        raise ValueError(f"Model {model_path} not supported")


def logits_to_text(logits, tokenizer):
    ids = logits.argmax(-1)
    try:
        if ids.shape == torch.Size([1]):
            generated_text = tokenizer.decode(ids, skip_special_tokens=True)
        else:
            generated_text = tokenizer.batch_decode(ids, skip_special_tokens=True)[0]
    except:
        print(f"Did not work ids: {ids}")
        print(f"Did not work logits: {logits}")
        generated_text = "ERROR decoding the logits"
    return generated_text

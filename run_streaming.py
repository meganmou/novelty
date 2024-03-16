# Adapted from StreamingLLM run_streaming_llm.py

import warnings

warnings.filterwarnings("ignore")

import torch
import argparse
import json
import os
import time
import re
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
os.environ["TRANSFORMERS_CACHE"] = "/path/to/cache"

from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftConfig, PeftModel
from streaming_llm.utils import load, download_url, load_jsonl
from streaming_llm.enable_streaming_llm import enable_streaming_llm

@torch.no_grad()
def greedy_generate(model, tokenizer, input_ids, past_key_values, max_gen_len, output_file):
    outputs = model(
        input_ids=input_ids,
        past_key_values=past_key_values,
        use_cache=True
    )
    past_key_values = outputs.past_key_values
    pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
    # # Apply temperature scaling to logits
    # temperature = 1.2
    # logits = outputs.logits[:, -1, :] / temperature
    # pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
    generated_ids = [pred_token_idx.item()]
    pos = 0
    for _ in range(max_gen_len - 1):
        outputs = model(
            input_ids=pred_token_idx,
            past_key_values=past_key_values,
            use_cache=True,
        )
        past_key_values = outputs.past_key_values
        # Apply temperature scaling to logits
        # logits = outputs.logits[:, -1, :] / temperature
        # pred_token_idx = logits.argmax(dim=-1).unsqueeze(1)
        pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
        generated_ids.append(pred_token_idx.item())
        generated_text = (
            tokenizer.decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
                spaces_between_special_tokens=False,
            )
            .strip()
            .split(" ")
        )

        now = len(generated_text) - 1
        if now > pos:
            # with open('output.txt', 'a') as f:
            #     print(" ".join(generated_text[pos:]), flush=True, file=f)
            print(" ".join(generated_text[pos:now]), end=" ", flush=True)
            pos = now

        if pred_token_idx == tokenizer.eos_token_id:
            break
    print(" ".join(generated_text[pos:]), flush=True)

    return past_key_values

@torch.no_grad()
def streaming_inference(model, tokenizer, prompts, kv_cache=None, max_gen_len=1500):
    past_key_values = None
    for idx, prompt in enumerate(prompts):
        prompt = "USER: " + prompt + "\n\nASSISTANT: "
        print("\n" + prompt, end="")
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        input_ids = input_ids.to(model.device)
        seq_len = input_ids.shape[1]
        if kv_cache is not None:
            space_needed = seq_len + max_gen_len
            past_key_values = kv_cache.evict_for_space(past_key_values, space_needed)

        past_key_values = greedy_generate(
            model, tokenizer, input_ids, past_key_values, max_gen_len=max_gen_len, output_file="output.txt"
        )
        # past_key_values = block_ngram_generate(
        #     model, tokenizer, input_ids, past_key_values, max_gen_len=max_gen_len, output_file="output.txt"
        # )


def main(args):
    # model_name_or_path = args.model_name_or_path
    # model, tokenizer = load(model_name_or_path)

    # LoRA config from peft model
    lora_config = PeftConfig.from_pretrained(args.peft_model_path)

    # Quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        lora_config.base_model_name_or_path,
        trust_remote_code=True,
    )

    # Load base model (within lora_config)
    model = AutoModelForCausalLM.from_pretrained(
        lora_config.base_model_name_or_path,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
        quantization_config=bnb_config,
    )

    # Load LoRA weights
    model = PeftModel.from_pretrained(model, args.peft_model_path)

    # Dynamic inference: user can input prompts and get responses
    # while True:
    #     user_input = input("USER: ")
    #     if user_input.lower() == 'quit':
    #         break  # Exit the loop if the user inputs 'quit'
    #     prompts = ["USER: " + user_input]
    #     # Perform inference on the input prompt
    #     if args.enable_streaming:
    #         kv_cache = enable_streaming_llm(
    #             model, start_size=args.start_size, recent_size=args.recent_size
    #         )
    #     else:
    #         kv_cache = None
    #     streaming_inference(model, tokenizer, prompts, kv_cache)

    test_filepath = "/data/all_vanilla_test_prompts.jsonl"
    print(f"Loading data from {test_filepath} ...")

    # if not os.path.exists(test_filepath):
    #     download_url(
    #         "https://raw.githubusercontent.com/lm-sys/FastChat/main/fastchat/llm_judge/data/mt_bench/question.jsonl",
    #         args.data_root,
    #     )
    #     os.rename(os.path.join(args.data_root, "question.jsonl"), test_filepath)

    list_data = load_jsonl(test_filepath)
    prompts = []
    for sample in list_data:
        prompts += sample["turns"]

    if args.enable_streaming:
        kv_cache = enable_streaming_llm(
            model, start_size=args.start_size, recent_size=args.recent_size
        )
    else:
        kv_cache = None

    streaming_inference(
        model,
        tokenizer,
        prompts,
        kv_cache,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path", type=str, default="lmsys/vicuna-7b-v1.5"
    )
    parser.add_argument(
        "--peft_model_path", type=str, default="/checkpoints/one-shot-finetuned-vicuna"
    )
    parser.add_argument("--data_root", type=str, default="data/")
    parser.add_argument("--enable_streaming", action="store_false")
    # parser.add_argument("--enable_streaming", action="store_true")
    parser.add_argument("--start_size", type=int, default=4)
    parser.add_argument("--recent_size", type=int, default=4000)
    args = parser.parse_args() 

    main(args)

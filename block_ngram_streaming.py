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
os.environ["TRANSFORMERS_CACHE"] = "/deep/group/aicc-bootcamp/vicuna"

from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftConfig, PeftModel
from streaming_llm.utils import load, download_url, load_jsonl
from streaming_llm.enable_streaming_llm import enable_streaming_llm

@torch.no_grad()
def block_ngram_generate(model, tokenizer, input_ids, past_key_values, max_gen_len, output_file, ngrams_to_store=10, ngram_length=3):
    outputs = model(
        input_ids=input_ids,
        past_key_values=past_key_values,
        use_cache=True,
    )
    past_key_values = outputs.past_key_values
    pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
    generated_ids = [pred_token_idx.item()]
    ngrams = set()
    pos = 0
    window = [generated_ids[-1]]
    for _ in range(max_gen_len - 1):
        outputs = model(
            input_ids=pred_token_idx,
            past_key_values=past_key_values,
            use_cache=True,
        )
        past_key_values = outputs.past_key_values
        logits = outputs.logits[:, -1, :]
        filtered_logits = filter_ngrams(logits, ngrams, tokenizer, ngram_length)
        pred_token_idx = filtered_logits.argmax(dim=-1).unsqueeze(1)
        generated_ids.append(pred_token_idx.item())
        window.append(generated_ids[-1])
        update_ngrams(window, ngrams, ngram_length, ngrams_to_store)
        if len(window) > ngrams_to_store:
            window.pop(0)

        now = len(window) - 1
        if now > pos:
            print(tokenizer.decode(window[pos:now]), end=" ", flush=True)
            pos = now

        if pred_token_idx == tokenizer.eos_token_id:
            break

    print(tokenizer.decode(window[pos:]), flush=True)

    with open(output_file, 'a') as f:
        f.write(tokenizer.decode(window[pos:]) + "\n")

    return past_key_values

@torch.no_grad()
def filter_ngrams(logits, ngrams, tokenizer, ngram_length):
    mask = torch.ones_like(logits) * -float('inf')
    for ng in ngrams:
        ng_ids = tokenizer.encode(ng)
        if len(ng_ids) > 1 and len(ng_ids) == ngram_length:
            ng_tensor = torch.tensor(ng_ids[:-1], device=logits.device)
            mask[:, ng_tensor] = 0
    return logits + mask

@torch.no_grad()
def update_ngrams(window, ngrams, ngram_length, ngrams_to_store):
    if len(window) < ngram_length:
        return
    new_ngram = " ".join(map(str, window[-ngram_length:]))
    ngrams.add(new_ngram)
    if len(ngrams) > ngrams_to_store:
        old_ngram = " ".join(map(str, window[-ngram_length - 1:-1]))
        ngrams.remove(old_ngram)

@torch.no_grad()
def streaming_inference(model, tokenizer, prompts, kv_cache=None, max_gen_len=1000):
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

        past_key_values = block_ngram_generate(
            model, tokenizer, input_ids, past_key_values, max_gen_len=max_gen_len, output_file="output.txt"
        )

def main(args):
    # model_name_or_path = args.model_name_or_path
    # model, tokenizer = load(model_name_or_path)

    # PeftConfig from peft model
    base_model_path = args.base_model_path
    config = PeftConfig.from_pretrained(args.peft_model_path)
    
    # Quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model_name_or_path,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
        quantization_config=bnb_config,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        config.base_model_name_or_path,
        trust_remote_code=True,
    )

    model = PeftModel.from_pretrained(model, args.peft_model_path)

    test_filepath = "/deep/u/joycech/LLaVA/data/solve_repetition_prompt_examples.jsonl"
    print(f"Loading data from {test_filepath} ...")

    if not os.path.exists(test_filepath):
        download_url(
            "https://raw.githubusercontent.com/lm-sys/FastChat/main/fastchat/llm_judge/data/mt_bench/question.jsonl",
            args.data_root,
        )
        os.rename(os.path.join(args.data_root, "question.jsonl"), test_filepath)

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
        "--base_model_path", type=str, default="lmsys/vicuna-7b-v1.5"
    )
    parser.add_argument(
        "--peft_model_path", type=str, default="/deep/u/joycech/LLaVA/checkpoints/finetuned-vicuna"
    )
    parser.add_argument("--data_root", type=str, default="data/")
    parser.add_argument("--enable_streaming", action="store_true")
    # parser.add_argument("--enable_streaming", action="store_true")
    parser.add_argument("--start_size", type=int, default=4)
    parser.add_argument("--recent_size", type=int, default=2000)
    args = parser.parse_args()

    main(args)
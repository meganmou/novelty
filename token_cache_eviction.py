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
from datasets import load_dataset
from torch.nn import CrossEntropyLoss
from streaming_llm.kv_cache import StartRecentKVCache
from streaming_llm.utils import parse_args, load

@torch.no_grad()
def streaming_inference(model, tokenizer, prompts, kv_cache=None, max_gen_len=2500):
    output_dir = "/deep/u/joycech/LLaVA/not-uploaded/streaming_llm/results"
    os.makedirs(output_dir, exist_ok=True)
    f = open(f"{output_dir}/log.txt", "w")
    
    past_key_values = None
    nlls = []
    loss_fn = CrossEntropyLoss(reduction="none")
    device = "cuda"
    num_eval_tokens = 0
    for idx, prompt in enumerate(prompts):
        prompt = "USER: " + prompt + "\n\nASSISTANT: "
        print("\n" + prompt, end="")
        encodings = tokenizer(prompt, return_tensors="pt")
        seq_len = encodings.input_ids.size(1)
        pbar = tqdm(range(0, seq_len - 1))
        
        for idx in pbar:
            input_ids = encodings.input_ids[:, idx : idx + 1].to(device)
            with torch.no_grad():
                outputs = model(
                input_ids,
                past_key_values=past_key_values,
                use_cache=True,
                )
                logits = outputs.logits.view(-1, model.config.vocab_size)
                past_key_values = outputs.past_key_values
                label = encodings.input_ids[:, idx + 1 : idx + 2].to(logits.device).view(-1)
                neg_log_likelihood = loss_fn(logits, label)
                if kv_cache is not None:
                    past_key_values = kv_cache(past_key_values)
            nlls.append(neg_log_likelihood)
            pbar.set_description(
                f"nll: {neg_log_likelihood.item():.2f}, ppl: {torch.exp(neg_log_likelihood).item():.2f}"
            )
            print(neg_log_likelihood.item(), file=f, flush=True)
            num_eval_tokens += 1
            if args.num_eval_tokens is not None and num_eval_tokens >= args.num_eval_tokens:
                break
        if args.num_eval_tokens is not None and num_eval_tokens >= args.num_eval_tokens:
            break
    
    f.close()

    ppl = torch.exp(torch.stack(nlls).mean())
    print(ppl.item())
    with open(f"{output_dir}/ppl.txt", "w") as f:
        f.write(f"{ppl.item()}\n")

def main(args):
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

    test_filepath = "/deep/u/joycech/LLaVA/data/example_test_instructions.jsonl"
    print(f"Loading data from {test_filepath} ...")

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
        "--peft_model_path", type=str, default="/deep/u/joycech/LLaVA/checkpoints/full-data-finetuned-vicuna"
    )
    parser.add_argument("--data_root", type=str, default="data/")
    parser.add_argument("--enable_streaming", action="store_true")
    # parser.add_argument("--enable_streaming", action="store_true")
    parser.add_argument("--start_size", type=int, default=4)
    parser.add_argument("--recent_size", type=int, default=4000)
    args = parser.parse_args() 

    main(args)
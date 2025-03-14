import collections
import datasets
import evaluate
import numpy as np
import torch
import torch.utils.benchmark as benchmark
from torch import nn
import transformers
from transformers import Trainer, TrainingArguments
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, LoraConfig
from trl import SFTTrainer
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
import argparse
import time
from transformers import LlamaForCausalLM
from lib.modelutils import *
from lib.datautils import *
from lib.prune import *
torch.manual_seed(100)

def get_llama(model):
    import torch
    def skip(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    model = LlamaForCausalLM.from_pretrained(
        model, 
        torch_dtype= 'auto', 
        device_map = "auto"
    )
    model.seqlen = model.config.max_position_embeddings 
    return model

if __name__ == "__main__":

    # Argument parser
    parser = argparse.ArgumentParser(description="Model fine-tuning script with pruning and quantization")
    parser.add_argument('--model', type=str, default="meta-llama/Llama-2-7b-hf", help='Name or path of the pre-trained model')
    parser.add_argument('--prune_method', type=str, default='magnitude', help='Method to use for pruning')

    parser.add_argument("--sparsity", type=float, default=0, help="Target sparsity") 
    parser.add_argument('--sparsity_ratio', type=float, default=0, help='Sparsity level')
    parser.add_argument("--sparsity_type", type=str, default="unstructured" ,choices=["unstructured", "4:8", "2:4"])
    parser.add_argument('--nsamples', type=int, default=128)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--dataset', type=str, default="wikitext2")
    parser.add_argument('--finetune_dataset', type=str, default="mlabonne/guanaco-llama2-1k")
    parser.add_argument('--pruned_model_dir', type=str, default="./pruned_model")
    parser.add_argument('--finetuned_model_dir', type=str, default="./finetuned_model")
    parser.add_argument('--peft_checkpoint_dir', type=str, default="./results/checkpoint-10")
    parser.add_argument('--quantized_model_dir', type=str, default="./quantized_model")

    args = parser.parse_args()

    model_name = args.model
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    dataset = args.dataset
    finetune_dataset = args.finetune_dataset
    pruned_model_dir = args.pruned_model_dir
    finetuned_model_dir = args.finetuned_model_dir
    peft_checkpoint_dir = args.peft_checkpoint_dir
    quantized_model_dir = args.quantized_model_dir
    prune_n, prune_m = 0, 0
    if args.sparsity_type != "unstructured":
        assert args.sparsity_ratio == 0.5, "sparsity ratio must be 0.5 for structured N:M sparsity"
        prune_n, prune_m = map(int, args.sparsity_type.split(":")) 

    # Get Model
    model = get_llama(model_name)
    model.eval()
    print("Pruning in process")
    
    # PRUNE
    if args.prune_method == "magnitude":
        model = prune_magnitude(args, model, tokenizer)
    elif args.prune_method == "sparsegpt":
        model = prune_sparsegpt(args, model, tokenizer, prune_n, prune_m)
    elif args.prune_method == "wanda":
         model = prune_wanda(args, model, tokenizer, prune_n = prune_n, prune_m = prune_m)
    else:
        print("ERROR: Invalid prune method. Choice=[\"magnitude\", \"sparsegpt\", \"wanda\"]")
        sys.exit(1)
    
    print("Pruning completed")
    
     
    save_model_and_tokenizer(model, tokenizer, pruned_model_dir)  
    
    
    model = LlamaForCausalLM.from_pretrained(
        args.pruned_model_dir,
        torch_dtype= torch.float16,
        device_map = "cuda:0"
    )
    model.eval()
    model.seqlen = 2048
    # Evaluate
    dataloader, testloader = get_loaders(
        args.dataset, seed=args.seed, model=args.model, seqlen=model.seqlen
    )
    print("Dataset:", args.dataset)
    llama_eval(model, testloader, DEV, args.dataset)
    
    
    
    # QUANTIZE
    print("Quantization in process")
    # Apply AUTOgptq
    quantize_config = BaseQuantizeConfig(
        bits=4,  # quantize model to 4-bit
        group_size=128,
        desc_act=False,  # set to False can significantly speed up inference but the perplexity may slightly bad
    )
    model = AutoGPTQForCausalLM.from_pretrained(pruned_model_dir, quantize_config)
    examples = [
        tokenizer(
            "auto-gptq is an easy-to-use model quantization library with user-friendly apis, based on GPTQ algorithm."
        )
    ]
    
    model.quantize(examples)
    print("Quantization completed")
    model.save_quantized(quantized_model_dir)
    
    
    # Evaluate Final quantized and pruned model
    model = AutoGPTQForCausalLM.from_quantized(
        quantized_model_dir,
        use_marlin=True,
        device="cuda:0"
    )
    model.seqlen = 2048
    model.eval()
    # Evaluate
    dataloader, testloader = get_loaders(
        args.dataset, seed=args.seed, model=args.model, seqlen=model.seqlen
    )
    print("Dataset:", args.dataset)
    llama_eval_wanda(model, testloader, DEV, args.dataset)



import torch 
import torch.nn as nn 
from .modelutils import *
from .datautils import *
from .sparsegpt import SparseGPT 
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from torch.ao.pruning import WeightNormSparsifier
from peft import LoraConfig, PeftModel
from trl import SFTTrainer
from datasets import load_dataset


def finetune_model(model_dir, dataset_name, output_dir="./results"):
    # Set up dtype and quantization configuration
    torch_dtype = torch.bfloat16
    compute_dtype = getattr(torch, "float16")
    
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=False,
    )

    # Load the model with quantization
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        quantization_config=quant_config,
        device_map={"": 0},
        low_cpu_mem_usage=True
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    print("Model loaded")

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Define PEFT parameters
    peft_params = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        bias="none",
        task_type="CAUSAL_LM",
    )
    # Define training arguments
    training_params = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=1,
        max_steps=10,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        optim="paged_adamw_8bit",
        warmup_steps=0.03,
        learning_rate=2e-4,
        fp16=True,
        save_steps=10,
        logging_steps=1000,
        push_to_hub=False,
        group_by_length=True,
        report_to='tensorboard',
    )

    # Load the Guanaco dataset
    dataset = load_dataset(dataset_name, split="train")

    # Initialize the SFTTrainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        # peft_config=peft_params,
        dataset_text_field="text",
        max_seq_length=256,
        tokenizer=tokenizer,
        args=training_params,
        packing=False,
    )
    trainer.neftune_noise_alpha = None

    trainer.train()
    print("Finetune finished")

def load_finetuned_model(base_model_id, peft_checkpoint_dir):
    # Define compute dtype
    compute_dtype = getattr(torch, "float16")

    # Load the base LLaMA model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        low_cpu_mem_usage=True,
        return_dict=True,
        torch_dtype=compute_dtype,
    )
    base_model.seqlen = base_model.config.max_position_embeddings 

    # Load the fine-tuned model with PEFT
    model = PeftModel.from_pretrained(base_model, peft_checkpoint_dir)
    model = model.merge_and_unload()  # Merge adapters and unload them from the model
    print("Fine-tuned model loaded and adapters merged.")
    model = AutoGPTQForCausalLM.from_pretrained(peft_checkpoint_dir)
    return model

def prune_magnitude(args, model, tokenizer):
    # Define the sparsifier for 2:4 sparsity
    sparsifier = WeightNormSparsifier(
        sparsity_level=1.0,  # 100% of blocks will be pruned to 2:4
        sparse_block_shape=(1, 4),  # Shape for 2:4 sparsity
        zeros_per_block=2  # Prune 2 out of every 4 elements in each block
    )

    # Apply sparsity to all nn.Linear layers except for the final output layer
    sparse_config = [
        {"tensor_fqn": f"{fqn}.weight"}
        for fqn, module in model.named_modules()
        if isinstance(module, nn.Linear)
    ]

    # Prepare and apply the sparsity
    sparsifier.prepare(model, sparse_config)
    sparsifier.step()
    # sparsifier.squash_mask()

    save_model_and_tokenizer(model, tokenizer, args.sparsified_model_dir)

    # Finetune model
    finetune_model(args.sparsified_model_dir, args.finetune_dataset, "./results")

    # Load, merge and save the fine-tuned model
    model = load_finetuned_model(args.model, args.peft_checkpoint_dir)
    # Squash model
    sparsifier.squash_mask()
    save_model_and_tokenizer(model, tokenizer, args.finetuned_model_dir)
    return model



    
@torch.no_grad()
def prune_sparsegpt(args, model, tokenizer):
    print("Starting...")
    dataloader, _ = get_loaders(
        args.dataset, nsamples=args.nsamples, seed=args.seed, model=args.model, seqlen=model.seqlen
    )
    dev = torch.device('cuda:0')
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {"i": 0, "attention_mask": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache["attention_mask"]

    print("Ready.")

    quantizers = {}
    for i in range(len(layers)):
        layer = layers[i].to(dev)
        full = find_layers(layer)
        sequential = [list(full.keys())]

        for names in sequential:
            subset = {n: full[n] for n in names}

            gpts = {}
            for name in subset:
                gpts[name] = SparseGPT(subset[name])

            def add_batch(name):
                def tmp(_, inp, out):
                    gpts[name].add_batch(inp[0].data, out.data)

                return tmp

            handles = []
            for name in subset:
                handles.append(subset[name].register_forward_hook(add_batch(name)))
            for j in range(args.nsamples):
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
            for h in handles:
                h.remove()

            for name in subset:
                print(i, name)
                print("Pruning ...")
                sparsity = args.sparsity
                gpts[name].fasterprune(
                    sparsity,
                    prunen=args.prunen,
                    prunem=args.prunem,
                    percdamp=0.01,
                    blocksize=128,
                )
                gpts[name].free()

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]

        layers[i] = layer.cpu()
        del layer
        del gpts
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache

    return quantizers

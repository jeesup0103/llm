import torch 
import torch.nn as nn 
from .modelutils import *
from .datautils import *
from .sparsegpt import SparseGPT 
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from torch.ao.pruning import WeightNormSparsifier
from peft import LoraConfig, PeftModel
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset
from .layerwrapper import WrappedGPT


def prepare_calibration_input(model, dataloader, device):
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers
    
    if "model.embed_tokens" in model.hf_device_map:
        device = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((128, model.seqlen, model.config.hidden_size), dtype=dtype, device=device)
    inps.requires_grad = False
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(device))
        except ValueError:
            pass 
    layers[0] = layers[0].module

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']
    model.config.use_cache = use_cache

    return inps, outs, attention_mask, position_ids 

def return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before):
    thres_cumsum = sum_before * alpha 
    sort_mask = tmp_metric <= thres_cumsum.reshape((-1,1))
    thres = torch.gather(sort_res[0], dim=1, index=sort_mask.sum(dim=1, keepdims=True)-1)
    W_mask = (W_metric <= thres)
    cur_sparsity = (W_mask==True).sum() / W_mask.numel()
    return W_mask, cur_sparsity


def finetune_model(model_dir, dataset_name, output_dir="./results"):
    # Set up dtype and quantization configuration
    torch_dtype = torch.bfloat16
    compute_dtype = getattr(torch, "float16")
    
    '''
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=False,
    )
    '''

    # Load the model with quantization
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.float16,
        #quantization_config=quant_config,
        device_map= "auto",
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
    '''
    # Define PEFT parameters
    peft_params = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        bias="none",
        task_type="CAUSAL_LM",
    )
    '''
    # Define training arguments
    training_params = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=1,
        max_steps=20,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=2e-4,
        # fp16=False,
        save_steps=10,
        group_by_length=True,
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
    return model

def prune_magnitude_llama(args, model, tokenizer):
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
    
    save_model_and_tokenizer(model, tokenizer, args.pruned_model_dir)

    # Finetune model
    finetune_model(args.pruned_model_dir, args.finetune_dataset, "./results")

    # Load, merge and save the fine-tuned model
    model = load_finetuned_model(args.model, args.peft_checkpoint_dir)
    # Squash model
    sparsifier.squash_mask()
    save_model_and_tokenizer(model, tokenizer, args.finetuned_model_dir)
    return model

def prune_magnitude_opt(args, model, tokenizer):
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
    
    save_model_and_tokenizer(model, tokenizer, args.pruned_model_dir)

    # Finetune model
    finetune_model(args.pruned_model_dir, args.finetune_dataset, "./results")
    # Load, merge and save the fine-tuned model
    model = load_finetuned_model(args.model, args.peft_checkpoint_dir)
    # Squash model
    sparsifier.squash_mask()
    save_model_and_tokenizer(model, tokenizer, args.finetuned_model_dir)
    return model

    
@torch.no_grad()
def prune_sparsegpt_llama(args, model, tokenizer, prune_n=0, prune_m=0):
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
                    prunen=prune_n,
                    prunem=prune_m,
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

    return model


   
@torch.no_grad()
def prune_sparsegpt_opt(args, model, tokenizer, prune_n=0, prune_m=0):
    print("Starting...")
    dataloader, _ = get_loaders(
        args.dataset, nsamples=args.nsamples, seed=args.seed, model=args.model, seqlen=model.seqlen
    )
    dev = torch.device('cuda:0')
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.decoer.layers

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
                    prunen=prune_n,
                    prunem=prune_m,
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

    return model

def prune_wanda_llama(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    use_cache = model.config.use_cache
    model.config.use_cache = False

    print("loading calibdation data")
    dataloader, _ = get_loaders(
        args.dataset, nsamples=args.nsamples, seed=args.seed, model=args.model, seqlen=model.seqlen
    )
    print("dataset loading complete")
    with torch.no_grad():
        inps, outs, attention_mask, position_ids = prepare_calibration_input(model, dataloader, device)

    layers = model.model.layers
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        for h in handles:
            h.remove()

        for name in subset:
            print(f"pruning layer {i} name {name}")
            W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))

            W_mask = (torch.zeros_like(W_metric) == 1)  ## initialize a mask to be all False
            if prune_n != 0:
                # structured n:m sparsity
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:,ii:(ii+prune_m)].float()
                        W_mask.scatter_(1,ii+torch.topk(tmp, prune_n,dim=1, largest=False)[1], True)
            else:
                sort_res = torch.sort(W_metric, dim=-1, stable=True)

                # unstructured pruning
                indices = sort_res[1][:,:int(W_metric.shape[1]*args.sparsity_ratio)]
                W_mask.scatter_(1, indices, True)

            subset[name].weight.data[W_mask] = 0  ## set weights to zero

        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        inps, outs = outs, inps

    model.config.use_cache = use_cache

    return model


def prune_wanda_opt(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    use_cache = model.config.use_cache
    model.config.use_cache = False

    print("loading calibdation data")
    dataloader, _ = get_loaders(
        args.dataset, nsamples=args.nsamples, seed=args.seed, model=args.model, seqlen=model.seqlen
    )
    print("dataset loading complete")
    with torch.no_grad():
        inps, outs, attention_mask, position_ids = prepare_calibration_input(model, dataloader, device)

    layers = model.model.decoders.layers
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        for h in handles:
            h.remove()

        for name in subset:
            print(f"pruning layer {i} name {name}")
            W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))

            W_mask = (torch.zeros_like(W_metric) == 1)  ## initialize a mask to be all False
            if prune_n != 0:
                # structured n:m sparsity
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:,ii:(ii+prune_m)].float()
                        W_mask.scatter_(1,ii+torch.topk(tmp, prune_n,dim=1, largest=False)[1], True)
            else:
                sort_res = torch.sort(W_metric, dim=-1, stable=True)

                # unstructured pruning
                indices = sort_res[1][:,:int(W_metric.shape[1]*args.sparsity_ratio)]
                W_mask.scatter_(1, indices, True)

            subset[name].weight.data[W_mask] = 0  ## set weights to zero

        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        inps, outs = outs, inps

    model.config.use_cache = use_cache

    return model

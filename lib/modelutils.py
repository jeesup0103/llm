import torch
import torch.nn as nn
import os

DEV = torch.device('cuda:0')


def find_layers(module, layers=[nn.Conv2d, nn.Linear], name=''):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res

def save_model_and_tokenizer(model, tokenizer, save_directory="./sparsified_model"):
    # Create the directory if it doesn't exist
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    # Save the model
    model.save_pretrained(save_directory)

    # Save the tokenizer
    tokenizer.save_pretrained(save_directory)

    print(f"Model and tokenizer saved to {os.path.abspath(save_directory)}")


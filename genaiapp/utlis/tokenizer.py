import os
import re
import sys

from transformers import AutoTokenizer


def Autotokenizer_data(model_name="bert-base-cased"):
    return AutoTokenizer.from_pretrained(model_name)

def batch_encode(texts, tokenizer, max_length=128):
    return tokenizer(
        texts,
        padding= True,
        truncation= True,
        max_length=max_length,
        return_tensors ="pt"
    )





from genaiapp.components.data_ingestion import load_movie_data
from genaiapp.utlis.tokenizer import Autotokenizer_data, batch_encode
import torch

plots = load_movie_data("genaiapp/data/wiki_movie_plots_deduped.csv")

tokenizer = Autotokenizer_data("bert-base-cased")

tokenized = batch_encode(plots[:10], tokenizer)

print(tokenized["input_ids"].shape)
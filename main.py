from genaiapp.components.data_ingestion import load_movie_data
from genaiapp.utlis.tokenizer import Autotokenizer_data, batch_encode
from genaiapp.utlis.batcher import create_batcher
import torch

plots = load_movie_data("genaiapp/data/wiki_movie_plots_deduped.csv")

tokenizer = Autotokenizer_data("bert-base-cased")

tokenized = batch_encode(plots[:10], tokenizer)

torch.save(tokenized['input_ids'], 'genaiapp/data/tokenized_input_id.pt')

batcher = create_batcher(input_ids= tokenized['input_ids'], batch_size=10)

print(tokenized["input_ids"].shape)
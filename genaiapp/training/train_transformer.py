
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from transformers import AutoModelForCausalLM , Trainer, TrainingArguments
from genaiapp.components.data_ingestion import load_movie_data
from genaiapp.utlis.tokenizer import Autotokenizer_data, batch_encode
from genaiapp.utlis.batcher import create_batcher

import torch
from dotenv import load_dotenv
load_dotenv()





def train():
    plots = load_movie_data('genaiapp/data/wiki_movie_plots_deduped.csv')
    tokenizer = Autotokenizer_data("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    encodings= batch_encode(plots, tokenizer, max_length=128)

    class TextDataset(torch.utils.data.Dataset):
        def __init__(self, encodings):
            self.encodings = encodings
        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item["labels"] = item["input_ids"].clone()
            return item
        def __len__(self):
            return len(self.encodings['input_ids'])
        
    dataset= TextDataset(encodings)

    model = AutoModelForCausalLM.from_pretrained("gpt2")

    training_arg = TrainingArguments(
        output_dir='./results',
        per_device_train_batch_size=10,
        num_train_epochs=2,
        save_steps=500,
        logging_steps=100,
        save_total_limit=2,
        prediction_loss_only=True
    )

    trainer = Trainer(
        model=model,
        args=training_arg,
        train_dataset=dataset
    )

    trainer.train()

    model.save_pretrained('./my_finetuned_LLM')
    tokenizer.save_pretrained('./my_finetuned_LLM')

if __name__=="__main__":
    train()
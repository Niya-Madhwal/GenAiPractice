import os
import sys
import numpy as np
import pandas as pd
from dotenv import load_dotenv
load_dotenv()


def load_movie_data(csv_path: str, text_column : str ="Plot") -> list:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"File not found on the defined path: {csv_path}")
    df = pd.read_csv(csv_path)
    if text_column not in df.columns:
        raise ValueError(f"column doesn't exist :{text_column}")
    plots= df[text_column].dropna().astype(str).tolist()
    return plots

if __name__ == '__main__':
    movie_plot= load_movie_data('..\data\wiki_movie_plots_deduped.csv')
    print(f"Loaded len{movie_plot}")
    print(movie_plot[0][:300])
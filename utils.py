# utils.py

import pandas as pd
import os

def load_data(directory):
    # Load data from IMDb Movie Reviews dataset
    data = []
    for category in ['pos', 'neg']:
        category_dir = os.path.join(directory, 'train', category)
        for filename in os.listdir(category_dir):
            with open(os.path.join(category_dir, filename), 'r', encoding='utf-8') as file:
                text = file.read()
            data.append({'text': text, 'sentiment': category})
    return pd.DataFrame(data)

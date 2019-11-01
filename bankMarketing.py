from pathlib import Path
import pandas as pd

# Set data path
DATA_DIR = Path('data/bankmarketing')

# Read csv to df
bank = pd.read_csv(DATA_DIR/'bank-additional-full.csv', delimiter=';')
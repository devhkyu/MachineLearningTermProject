from pathlib import Path
from sklearn.utils import resample
import pandas as pd

# Set Directory and read file
DATA_DIR = Path('../data/world_development_indicators/pivot_indicator')
SAVE_DIR = Path('../data/world_development_indicators/prophet')
filename = 'Health-Energy-Education.csv'
df = pd.read_csv(DATA_DIR/filename)

# Split columns as you want
# col = df.columns
# new = pd.DataFrame({col[0]: df[col[0]], col[1]: df[col[1]], col[2]: df[col[2]]})
# new = resample(new, replace=True, n_samples=int(2000), random_state=123)

# Resampling
new = resample(df, replace=True, n_samples=int(1000), random_state=123)

# Save to csv
new.to_csv(SAVE_DIR/filename)
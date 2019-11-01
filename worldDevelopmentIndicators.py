from pathlib import Path
import pandas as pd

# Set data path
DATA_DIR = Path('data/world_development_indicators')

# Read csv to df
country = pd.read_csv(DATA_DIR/'Country.csv', delimiter=',')
countryNotes = pd.read_csv(DATA_DIR/'CountryNotes.csv', delimiter=',')
footNotes = pd.read_csv(DATA_DIR/'Footnotes.csv', delimiter=',')
indicators = pd.read_csv(DATA_DIR/'Indicators.csv', delimiter=',')
series = pd.read_csv(DATA_DIR/'Series.csv', delimiter=',')
seriesNotes = pd.read_csv(DATA_DIR/'SeriesNotes.csv', delimiter=',')
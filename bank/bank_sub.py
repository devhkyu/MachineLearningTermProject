from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
import pandas as pd
import matplotlib.pyplot as plt

# Set data path
DATA_DIR = Path('../data/bankmarketing')

# Read csv to df
bank = pd.read_csv(DATA_DIR/'bank-additional-full.csv', delimiter=';')
label_encoder = LabelEncoder()

x = bank[['age', 'duration']]
y = label_encoder.fit_transform(bank['y'])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1, stratify=y)

knn = KNeighborsClassifier(n_neighbors=2)
knn.fit(x_train, y_train)
score = knn.score(x_test, y_test)
print(score)

plt.scatter(label_encoder.fit_transform(bank['month']), bank['duration'])
plt.title('month-duration')
plt.xlabel('month')
plt.ylabel('duration')
plt.show()

plt.scatter(label_encoder.fit_transform(bank['contact']), bank['duration'])
plt.title('contact-duration')
plt.xlabel('contact')
plt.ylabel('duration')
plt.show()
import pandas as pd
import pickle
from xgboost import XGBClassifier

df = pd.read_csv('./data/features/train.csv')

x = df.drop(columns=['income >50K'], axis=1)
y = df['income >50K']

model = XGBClassifier()
model.fit(x, y)

pickle.dump(model, open('model.pkl', 'wb'))
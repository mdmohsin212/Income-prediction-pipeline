import pandas as pd
import json
import pickle
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score, roc_auc_score

model = pickle.load(open('model.pkl', 'rb'))
test_data = pd.read_csv('./data/features/test.csv')

x_test = test_data.drop(columns=['income >50K'], axis=1)
y_test = test_data['income >50K']

y_pred = model.predict(x_test)
y_pred_proba = model.predict_proba(x_test)[:,1]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)

metrics_dict = {
    'accuracy' : accuracy,
    'precision' : precision,
    'recall' : recall,
    'f1' : f1,
    'auc' : auc
}

with open('metrics.json', 'w') as file:
    json.dump(metrics_dict, file, indent=4)
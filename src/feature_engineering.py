import os
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler

train_data = pd.read_csv('./data/process/Train_process.csv')
test_data = pd.read_csv('./data/process/Test_process.csv')

X = train_data.drop('income >50K', axis=1)
y = train_data['income >50K']

smt = SMOTE()
x_train, y_train = smt.fit_resample(train_data.drop('income >50K', axis=1), train_data['income >50K'])

enc = StandardScaler()
train_X_scaled = enc.fit_transform(x_train)
test_X_scaled = enc.transform(test_data.drop('income >50K', axis=1))

path = os.path.join('data', 'features')
os.makedirs(path, exist_ok=True)

train_X_scaled_df = pd.DataFrame(train_X_scaled, columns=X.columns)
train_Y_scaled_df = pd.DataFrame(y_train, columns=['income >50K'])
train_df = pd.concat([train_X_scaled_df, train_Y_scaled_df], axis=1)

test_X_scaled_df = pd.DataFrame(test_X_scaled, columns=X.columns)
test_df = pd.concat([test_X_scaled_df, test_data['income >50K']], axis=1)

train_df.to_csv(os.path.join(path, 'train.csv'), index=False)
test_df.to_csv(os.path.join(path, 'test.csv'), index=False)
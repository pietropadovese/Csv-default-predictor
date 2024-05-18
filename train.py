import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import  LogisticRegression
from sklearn.pipeline import Pipeline

data = pd.read_csv("data/data.csv")

X = data.iloc[:,1:-1]
y = data.iloc[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = LogisticRegression(random_state=0).fit(X_train, y_train)

pickle.dump(clf, open('model.pkl', 'wb'))
import pandas as pd
from pyexpat import features
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import numpy as np

train_data = pd.read_csv('../Data/train.csv')
test_data = pd.read_csv('../Data/test.csv')
train_data.info()
train_data.describe()
train_data.isnull().sum()

features = ['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']
train = train_data[features+['Survived']].copy()
print(train.columns)

train['Age'] = train['Age'].fillna(train['Age'].median())  # Fill missing ages with median
train['Embarked'] = train['Embarked'].fillna(train['Embarked'].mode()[0])
test = test_data[features].copy()
test['Age']= test['Age'].fillna(train['Age'].median())
test['Fare']= test['Fare'].fillna(train['Fare'].median())
test['Embarked'] = test['Embarked'].fillna(train['Embarked'].mode()[0])

train = pd.get_dummies(train,columns=['Sex','Embarked'],drop_first=True)
x_train = train.drop('Survived',axis=1)
y_train = train['Survived']
test = pd.get_dummies(test, columns=['Sex', 'Embarked'], drop_first=True)
X_test = test
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(x_train, y_train)
predictions = model.predict(X_test)
importances = model.feature_importances_
feature_names = x_train.columns
feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Create visualization
sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
plt.title('Feature Importance - Which factors most influence survival?')
plt.show()

joblib.dump(x_train.columns.tolist(), "rf_model_titanic_columns.joblib")
joblib.dump(model, 'rf_model_titanic.joblib')
print("Model saved successfully!")
print("teste")
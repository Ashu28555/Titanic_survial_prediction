# codealpha_tasks
# Task 1
## Make a system which tells whether the person will be save from sinking. What factors were most likely lead to success-socio-economic status, age, gender and more.

# Importing the Library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

# Importing Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

## Load the dataset

titanic_data = pd.read_csv('Titanic-Dataset.csv')

titanic_data.head()

## Preprocessing of data

titanic_data = titanic_data.drop(['Name', 'Ticket', 'Fare','Cabin','PassengerId'], axis=1)

titanic_data.head()

titanic_data['Sex'].value_counts()

titanic_data['Embarked'].value_counts()

titanic_data['Survived'].value_counts()

titanic_data['Sex'] = titanic_data['Sex'].map({'male':0,'female':1})
titanic_data['Embarked'] = titanic_data['Embarked'].map({'S':0,'C':1,'Q':2})
titanic_data['FamilySize'] = titanic_data['SibSp'] + titanic_data['Parch']

titanic_data.head()

titanic_data['FamilySize'].value_counts()

titanic_data.shape

titanic_data.info()

titanic_data.describe()

titanic_data.dropna(inplace=True)

titanic_data

## Exploratory Data Analysis

survived_data = titanic_data[titanic_data['Survived'] == 1]

sns.countplot(x='Sex', data=survived_data)
plt.title('Survival Count by Gender')
plt.xlabel('Gender')
plt.ylabel('Survival Count')
plt.xticks([0, 1], ['Male', 'Female']) 
plt.show()

survived_data = titanic_data[titanic_data['Survived'] == 1]

sns.countplot(x='FamilySize', data=survived_data)
plt.title('Survival Count by Family Size')
plt.xlabel('Family Size')
plt.ylabel('Survival Count')
plt.show()

titanic_data['FamilySize'].value_counts()

titanic_data

titanic_data = titanic_data.drop(['SibSp','Parch'], axis=1)

x = titanic_data.drop(['Survived'], axis=1)

x

y = titanic_data['Survived']

y

## Splitting the Data set into testing and training Dataset

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42,stratify=y)

x_train

x_test

## Model Evaluation by Cross Validation

models=[LogisticRegression(max_iter=10000), SVC(), KNeighborsClassifier(),RandomForestClassifier(random_state=42)]

def compare_models_cross_validation():
     for model in models:
            cv_score=cross_val_score(model, x, y, cv=6)
            mean_accuracy=sum(cv_score)/len(cv_score)
            mean_accuracy=mean_accuracy*100
            mean_accuracy=round(mean_accuracy, 2)
            print('Cross Validation accuracies for the',model,'=', cv_score)
            print('Acccuracy score of the ',model,'=',mean_accuracy,'%')
            print('---------------------------------------------------------------')

compare_models_cross_validation()

### Use the Logistic Regression Model for this data set because it give good accuracy from others. Now, we apply Hyperparameter tuning by Grid search CV

## Using Hyper Parameter tuning by Grid Search CV

param_grid = {
    'C' : [0.001,0.01,0.1,1,10,100]
}

model1 = LogisticRegression(max_iter = 1000)

grid_search = GridSearchCV(estimator = model1,param_grid=param_grid,cv=5,scoring='accuracy')

grid_search.fit(x_train,y_train)

grid_search.best_params_

grid_search.best_score_

y_pred = grid_search.predict(x_test)
y_pred

cm = confusion_matrix(y_pred,y_test)

cm

ConfusionMatrixDisplay(cm,display_labels=['Not Survived','Survived']).plot()

report = classification_report(y_pred,y_test)
print(report)

# Model to predict whether the person will be save from sinking.

def predict_survival(model):
    sex = input("Enter gender (male/female): ").lower()
    age = float(input("Enter age: "))
    pclass = int(input("Enter passenger class (1, 2, or 3): "))
    embarked = input("Enter embarked port (S, C, or Q): ").upper()
    family_size = int(input("Enter family size: "))
    
    sex = 0 if sex == 'male' else 1
    embarked = {'S': 0, 'C': 1, 'Q': 2}[embarked]
    
    # Create DataFrame with consistent order of features
    input_data = pd.DataFrame({'Sex': [sex], 'Age': [age], 'Pclass': [pclass],
                               'Embarked': [embarked], 'FamilySize': [family_size]})
    
    # Reorder columns to match training data
    input_data = input_data[['Pclass', 'Sex', 'Age', 'Embarked', 'FamilySize']]
    
    # Make prediction
    prediction = model.predict(input_data)
    
    if prediction[0] == 1:
        print("The model predicts that the passenger will survived.")
    else:
        print("The model predicts that the passenger will not survived.")

# Use the best model (grid_search) to make predictions
predict_survival(grid_search)


# The End


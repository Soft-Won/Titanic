import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

# Extract = ['Survived','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']
Extract = ['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']
x_law = pd.read_csv('/home/kang/Titanic/Code/input/test.csv')
extracted = x_law[Extract]

#Completing : Null Value Exit
extracted['Age'].fillna(extracted['Age'].median(), inplace = True)
extracted['Embarked'].fillna(extracted['Embarked'].mode()[0], inplace = True)
extracted['Fare'].fillna(extracted['Fare'].median(), inplace = True)

#Creating : Not

# Method 1 : Not Dummy
# #Converting : Str -> Code
label = LabelEncoder()
extracted['Sex'] = label.fit_transform(extracted['Sex'])
extracted['Embarked'] = label.fit_transform(extracted['Embarked'])

# Method 2 : Go Dummy
# extracted = pd.get_dummies(extracted[Extract])

extracted['Age'] = extracted['Age'].round(0).astype('int')
extracted['Fare'] = extracted['Fare'].round(0).astype('int')
#print(extracted['Survived'])

print(extracted)
# extracted.info()

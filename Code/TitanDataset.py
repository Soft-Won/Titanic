import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#%matplotlib inline

import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

class ToTensor(object):
    def __call__(self, pic):
        return F.to_tensor(pic)
    
    def __repr__(self):
        return self.__class__.__name__ + '()'

class Dataset(object):
    def __getitem__(self, index):
        raise NotImplementedError
    def __len__(self):
        raise NotImplementedError
    def __add__(self, other):
        return ConcatDataset([self, other])

class TitanDataset(Dataset):
    def __init__(self, file_path=None, transform=None, purpose='train'):
        self.purpose = purpose
        if purpose == 'train'   :
            Extract = ['Survived','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']
            x_law = pd.read_csv('/home/kang/Titanic/Code/input/train.csv')
        elif purpose == 'test'  :
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
        # label = LabelEncoder()
        # extracted['Sex'] = label.fit_transform(extracted['Sex'])
        # extracted['Embarked'] = label.fit_transform(extracted['Embarked'])

        # Method 2 : Go Dummy
        extracted = pd.get_dummies(extracted[Extract])
        
        extracted['Age'] = extracted['Age'].round(0).astype('int')
        extracted['Fare'] = extracted['Fare'].round(0).astype('int')

        self.data = extracted
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if self.purpose == 'train':
            X = self.data.iloc[index,1:].values.astype(np.float32)
            Y = self.data.iloc[index,0]
        elif self.purpose == 'test' : 
            X = self.data.iloc[index,0:].values.astype(np.float32)
            Y = 0
        return X,Y

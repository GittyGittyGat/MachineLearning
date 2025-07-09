import os
import pandas as pd
import numpy as np
from sklearn.feature_selection import r_regression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import math
from sklearn.ensemble import RandomForestRegressor

training_data_file_path = 'train.csv'
testing_data_file_path = 'test.csv'

training_data = pd.read_csv(training_data_file_path)
testing_data = pd.read_csv(testing_data_file_path)

# training_data.describe().to_csv('data_desc.csv')

# it is kind of hard to look at the data and understand which factors matter,
# let's take something abstract like MS Sub Class I have no idea what that means
# let alone how it affects the pricing of the home...
# and I can read up on it and figure it out, but why not let sklearn do it
# algorithmically. One way to do this is to use the SelectKBest() function
# but then I don't know how many factors matter, and I can run every single number
# but there's something better r_regression.
# from my understanding this finds how related two variables are, then we can eliminate ones
# that don't matter

features = ['MSSubClass',
'LotArea',
'OverallQual',
'OverallCond',
'YearBuilt',
'YearRemodAdd',
'1stFlrSF',
'2ndFlrSF',
'LowQualFinSF',
'GrLivArea',
'FullBath',
'HalfBath',
'BedroomAbvGr',
'KitchenAbvGr',
'TotRmsAbvGrd',
'Fireplaces',
'WoodDeckSF',
'OpenPorchSF',
'EnclosedPorch',
'3SsnPorch',
'ScreenPorch',
'PoolArea',
'MiscVal',
'MoSold',
'YrSold']

X = training_data[features]
y = training_data.SalePrice

r_corr = r_regression(X,y)

# looking at the array from r_regression, anything factor 
# with no statistical significance (less than .05)
# should be removed

strong_features = []

for i in range(len(r_corr)):
    if(abs(r_corr[i]) >= 0.2):
        strong_features.append(features[i])

# print(len(strong_features))

def evaluateTree(max_leaf_nodes, train_X, train_y, val_X, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    return mean_absolute_error(val_y, preds_val)

best_fit = 0
least_dif = math.inf

new_X = training_data[strong_features]
train_X, val_X, train_y, val_y = train_test_split(new_X, y, random_state=1)

# temp = 0
for i in range(2, 500):
    temp = evaluateTree(i, train_X, train_y, val_X, val_y)
    if(temp < least_dif):
                    # about 24256 -- not very good at .05, let's increase threshold
                    # at .10 it's 22804, let's see if increasing it more helps
                    # at .2 it's 22195, at .3 it's 22436 so idea value is between .1-.3
                    # 13 factors seems to give the best results - so let's stick with that!
        least_dif = temp
        best_fit = i

# print(temp)
# print(best_fit) # 41 is best fit (at .05)! -> with changes it became 101 (at .2 or 13 factors)!

# haha! this was worse than previous submission, let's try this with randomforests (13 factors)
# with random forests the prediction improved by about 4000$ and went down from 22.5k (first submission)
# to 18.3k, which is a slight improvement!

testing_X = testing_data[strong_features]

# final_model = DecisionTreeRegressor(max_leaf_nodes=best_fit, random_state = 0)

final_model = RandomForestRegressor(random_state=1)
final_model.fit(new_X, y)
predictions = final_model.predict(testing_X)
output = pd.DataFrame({'Id':testing_data.Id, 'SalePrice':predictions})
output.to_csv('submissionRF.csv', index=False)
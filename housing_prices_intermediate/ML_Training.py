import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import r_regression
from sklearn.metrics import mean_absolute_error
import numpy as np
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor
from sklearn.compose import ColumnTransformer
#from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

# the first thing I want to do is look at the data and decide what will be
# leakage (data I shouldn't use) data, what will be Ordinal Encoding and One-Hot Encoding 
# so let's get the datatypes of all the data

train_file_path = 'train.csv'
test_file_path = 'test.csv'

train_file = pd.read_csv(train_file_path)
test_file = pd.read_csv(test_file_path)

X = train_file.drop('SalePrice', axis=1)
X_test = test_file

missing_percentage = X.isna().mean()
columns_with_missing = missing_percentage[missing_percentage > 0.33].index
X = X.drop(columns_with_missing, axis=1)


y = train_file.SalePrice

numerical_data = [cname for cname in X.columns if X[cname].dtype in ['int64']]
float_data = [cname for cname in X.columns if X[cname].dtype in ['float64']]
categorical_data = [col for col in X.columns if X[col].dtype == "object"]

imputer = SimpleImputer(strategy='constant')
X_calculations = X[numerical_data]
imputer.fit_transform(X_calculations)

final_columns = []
final_nums = []
# figure out if numerical data is valuable and add it to our features list
for i in numerical_data:
    r_corr = r_regression(pd.DataFrame(X_calculations[i]), y)
    if(abs(r_corr[0]) > 0.05):
        final_columns.append(i)
        final_nums.append(i)

final_float = []
# figure out if float data is valuable and add it to our features list
for i in float_data:
    feature_data = pd.DataFrame(X[i]).astype(np.float64)
    imputer2 = SimpleImputer(strategy='constant')
    feature_data = imputer2.fit_transform(feature_data)
    r_corr = r_regression(feature_data, y, force_finite=False)
    if(abs(r_corr[0]) > 0.05):
        final_columns.append(i)
        final_float.append(i)

Ord_Enc = ["Neighborhood", "Exterior1st", "Exterior2nd"]
OH_Enc = list(set(categorical_data) - set(Ord_Enc))
final_OH = []
final_ord = []

for i in Ord_Enc:
    feature_data = pd.DataFrame(X[i])
    imputer2 = SimpleImputer(strategy='constant')
    feature_data = imputer2.fit_transform(feature_data)
    ord_enc = OrdinalEncoder()
    feature_data = pd.DataFrame(ord_enc.fit_transform(feature_data))
    r_corr = r_regression(feature_data, y)
    if(abs(r_corr[0]) > 0.05):
        final_columns.append(i)
        final_ord.append(i)

for i in OH_Enc:
    feature_data = pd.DataFrame(X[i])
    imputer2 = SimpleImputer(strategy='constant')
    feature_data = imputer2.fit_transform(feature_data)
    oh_enc = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    feature_data = pd.DataFrame(oh_enc.fit_transform(feature_data))
    r_corr = r_regression(feature_data, y)
    if(abs(r_corr[0]) > 0.05):
        final_columns.append(i)
        final_OH.append(i)



# actually before examining the data lets drop the data that 
# is inconsistent with the test data
"""categorical_test_data = [col for col in X_test.columns if X_test[col].dtype == "object"]

good_categories = [col for col in categorical_data if set(categorical_data[col.index(col)]).issubset(set(categorical_test_data[col.index(col)]))]
"""

#print(good_categories) all categorical categories are a subset of test data. 
"""
object_nunique = list(map(lambda col: X[col].nunique(), categorical_data))
d = dict(zip(categorical_data, object_nunique))

sorted(d.items(), key=lambda x: x[1])
for i in d.items():
    print(i)
"""
# Neighborhood (25), Exterior1st(15), Exterior2nd(16) should not be One-Hot

# next up is eliminiating variables that may cause contaimination
# I think there are two ways to do this: 
# method 1:
#   Read what every single column of data and figure out wehther it does possibly introduce contamination
# method 2:
#   Weed out suspicious columns of data by eliminating too high of a correlation (>90% -- I can change this and test it later)
#   and weeding out columns with too low of a correlation (those columns don't the sale price so we can just ignore them)
# step 1: would be to change the categorical data via One-Hot encoding or Ordinal Encoding
# I will set up pipelines later, but just to figure out what kind of data should be used I'm processing the data one by one
    
X_final = X[final_columns]
X_test_final = X_test[final_columns]

imputer1 = SimpleImputer(strategy='mean')

categorical_OH_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

categorical_Ord_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('ordinal', OrdinalEncoder())
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', imputer1, final_nums),
        ('num2', imputer1, final_float),
        ('cat', categorical_Ord_transformer, final_ord),
        ('ord', categorical_OH_transformer, final_OH)
])

my_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', XGBRegressor(learning_rate=0.05, n_estimators=550, random_state=0, n_jobs=4))])

"""scoresList = []
for i in range(50, 1500, 50):
    my_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', XGBRegressor(learning_rate=0.05, n_estimators=i, random_state=0, n_jobs=4))])
    scores = -1 * cross_val_score(my_pipeline, X_final, y,
                              cv=5,
                              scoring='neg_mean_absolute_error', n_jobs=5)
    print(i)
    scoresList.append(scores)

for i in range(len(scoresList)):
    print(i, scoresList[i])"""


my_pipeline.fit(X_final, y)

predictions = my_pipeline.predict(X_test)
output = pd.DataFrame({'Id':X_test.Id, 'SalePrice':predictions})
output.to_csv('submissionIM.csv', index=False)

# managed to score 421st on the leaderboard! Yay!!!

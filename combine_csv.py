import numpy as np
import pandas as pd
import pickle
from sklearn.impute import SimpleImputer

# DATA PREPROCESSING

datasets = pd.read_csv('full_combined.csv')

# STEP-1 : Selecting the required columns from the set of columns in dataset.
new_dataset = datasets.iloc[:,[8,9,10,11]]

# STEP-2 : Converted selected column data to csv file again.    
new_dataset.to_csv('final_dataset.csv')


# STEP-3: Replacing missing values from the cell by mean of the column values.
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')

imputer = imputer.fit(new_dataset[1:3])

new_dataset[1:3] = imputer.transform(new_dataset[1:3])

# As There is no repeating data in final_dataset we are not performing
# step which includes encoding of categorical data from data preprocessing.


#STEP-4 : decide and set input and output values for prediction

input_training_dateset = new_dataset[['dislikes', 'comment_count', 'views']].values
output_training_dateset = new_dataset[['likes']].values

from sklearn.model_selection import train_test_split


#STEP-5 : split the dataset into training dataset and testing dataset parts
#       Training dataset will be used to train the model and testing dataset will be used to check its prediction accuracy.

X_train, X_test, y_train, y_test = train_test_split(input_training_dateset, output_training_dateset, test_size=0.2, train_size = 0.8, random_state=0)

#Here we are creating one common datafile which will be used as a general datasource for all the algorithms that
#we want to implement on it.

pfile = open('train_test_data', 'wb')
pickle.dump(X_train, pfile)
pickle.dump(X_test, pfile)
pickle.dump(y_train, pfile)
pickle.dump(y_test, pfile)
pfile.close()



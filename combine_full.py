
import pandas as pdlib


# Combining data from all the excel files into single excel file

datasets = []
datasets.append(pdlib.read_csv('countrywise_Data/CANADA.csv'))
datasets.append(pdlib.read_csv('countrywise_Data/DENMARK.csv'))
datasets.append(pdlib.read_csv('countrywise_Data/FRANCE.csv'))
datasets.append(pdlib.read_csv('countrywise_Data/ENGLAND.csv'))
datasets.append(pdlib.read_csv('countrywise_Data/US.csv'))

new_dataset = datasets[0]

for dataset in datasets[1:]:
    new_dataset = new_dataset.append(dataset, ignore_index='true')
    
new_dataset.to_csv('full_combined.csv')


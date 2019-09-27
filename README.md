"""Read me file for MOD 1 Project"""

### REPO:
* Images - Folder containing all saved images rendered from code
* better_kd.csv - Final dataset with engineered features
* columns_name.md - An updated file contining the original feature names, as well as the engineered features
* CONTRIBUTING.md - No Clue
* kc_heatmap.html - Interactive Heat Map of kc_house_data 
* kc_house_data.csv - Original dataset, RAW
* kc_map.html - Interactive Map with Circle Markers of kc_house_data 
* LICENSE.md - No Clue
* new_kd - Final dataset without engineered features 
* presentation.pdf - Non_Technical presentation of findings
* README.md - This document
* student.ipynb - Technical document with code and model

### Introduction: Sell My Home (SMH)

For the SMH project, I worked with the King County House Sales dataset. I modified the dataset to make it a easy for potential home sellers to understand.  The original dataset can be found in the file `"kc_house_data.csv"`, in this repo.The description of the column names can be found in the `column_names.md` file in this repository. As with most real world data sets, the column names were not perfectly described, so I made a few features to make it a bit easier to interpret. I cleaned, explored, and modeled this dataset with a multivariate linear regression to predict the sales price of homes as accurately as possible.

### Cleaning

kd['date'] =  pd.to_datetime(kd['date'],
                              format='%m/%d/%Y')
                              
kd = kd.dropna(axis=0,subset=['view'])

kd = kd[kd.bedrooms <5]

kd = kd.drop(axis=1,labels=['waterfront', 'yr_renovated'])

new_kd = kd[columns_to_keep].copy() # making a copy of the DataFrame



### Exploring

display(kd.info())
kd.describe()
kd.isna().sum()

### Visual Exploration

new_kd.hist(figsize=(16,9));

from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
plt.style.use('seaborn')
%matplotlib inline

pd.plotting.scatter_matrix(new_kd, figsize=(16,9));
plt.show()

###

What I did
Datasets
Findings
Recommendation
To reproduce, here is how

###



### Modeling


#### Model Quality/Approach

* Your model should not include any predictors with p-values greater than .05.  
* Your notebook shows an iterative approach to modeling, and details the parameters and results of the model at each iteration.  
    * **Level Up**: Whenever necessary, you briefly explain the changes made from one iteration to the next, and why you made these choices.  
* You provide at least 1 paragraph explaining your final model.   
* You pick at least 3 coefficients from your final model and explain their impact on the price of a house in this dataset.   



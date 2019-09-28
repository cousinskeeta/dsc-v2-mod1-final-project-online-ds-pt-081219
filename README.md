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

For the SMH project, I worked with the King County House Sales dataset. I modified the dataset to make it a easy for potential home sellers to understand.  The original dataset can be found in the file `"kc_house_data.csv"`, in this repo.The description of the column names can be found in the `column_names.md` file in this repository. As with most real world data sets, the column names were not perfectly described, so I made a few features to make it a bit easier to interpret. I cleaned, explored, and modeled this dataset with a multivariate linear regression models to identify the best features is predicting the sales price of homes as accurately as possible.

### Where are houses selling in King County?

![Where Houses are Viewed the Most](https://github.com/cousinskeeta/dsc-v2-mod1-final-project-online-ds-pt-081219/blob/master/Images/Screen%20Shot%202019-09-25%20at%2011.41.24%20PM.png )

### What's the average price to sell in the KC housing market?

![Price By Bedrooms](https://github.com/cousinskeeta/dsc-v2-mod1-final-project-online-ds-pt-081219/blob/master/Images/Price%20by%20Bedrooms.png)

![Price By Bathrooms](https://github.com/cousinskeeta/dsc-v2-mod1-final-project-online-ds-pt-081219/blob/master/Images/Price%20by%20Bathrooms.png)


### What features correlate to higher home prices? 

![Bath/Bed Ratio By Price](https://github.com/cousinskeeta/dsc-v2-mod1-final-project-online-ds-pt-081219/blob/master/Images/Bed-Bath%20Prices.png)


### Importing and Exploring the Dataset

Libraries:

    import pandas as pd
    import numpy as np

Data_Cleaning:

    df = pd.read_csv("kc_house_data.csv")
    display(df.info())
    df.describe()
    df.isna().sum()
    df['date'] =  pd.to_datetime(df['date'],format='%m/%d/%Y')
    df = df.dropna(axis=0,subset=['view'])
    df = df[df.bedrooms <5]
    df = df.drop(axis=1,labels=['waterfront', 'yr_renovated'])


### Visual Exploration

Libraries:

    from pandas.plotting import scatter_matrix
    import matplotlib.pyplot as plt
    %matplotlib inline
    plt.style.use('seaborn')
    import seaborn as sns
    import folium 
    import folium.plugins as plugins
    
Visualization: 

    df.hist(figsize=(16,9));

    pd.plotting.scatter_matrix(df, figsize=(16,9));
    plt.show()

    sns.heatmap(df.corr(), center=0, fmt='.2g', annot=True);

    kc_hm = folium.Map(location=[lat, lon], zoom_start=10.25)
    kc_hm
    locations = list(lat, lon, hue)) 
    hm = plugins.HeatMap(df=locations, radius=8 , blur=4.8, overlay=True) 
    hm.add_to(kc_hm) 
    kc_hm.save('heatmap.html') 
    
### Linear model in Statsmodels - OLS 

Libraries: 

    from statsmodels.formula.api import ols

Ordinary Least Squares (OLS):

    outcome = 'price'
    features = ['sqft_living','bedrooms', 'bathrooms','floors', 
                'condition', 'view', 'yr_built',
                'zipcode','lat', 'long'] 
    predictors = '+'.join(features)
    formula = outcome + "~" + predictors
    model = ols(formula=formula, data=df).fit()
    model.summary()
    
R-squared:

    0.641

### Feature Engineering & RFE 

Engineered_Features:

    $/sqft = Price per Square Foot
    bath/bed = ratio of bathrooms to bedrooms
    bath/sqft = bathrooms per square foot of living
    bed/sqft = bedrooms per square foot of living

Libraries:

    from sklearn.feature_selection import RFE
    from sklearn.linear_model import LinearRegression
    
Recursive Features Elimination (RFE):

    linreg = LinearRegression()
    selector = RFE(linreg, n_features_to_select = 5) 
    selector = selector.fit(X, y.values.ravel())
    selected_columns = X.columns[selector.support_ ]
    linreg.fit(X[selected_columns],y)
    yhat = linreg.predict(X[selected_columns])
    SS_Residual = np.sum((y-yhat)^2)
    SS_Total = np.sum((y-np.mean(y))^2)
    r_squared = 1 - (float(SS_Residual))/SS_Total
    adjusted_r_squared = 1 - (1-r_squared)*(len(y)-1)/(len(y)-X[selected_columns].shape[1]-1)

R_squared:

    0.8922987434967452
    
Adjusted_R_squared:

    0.8922712686864127

### Train-Test-Split  / Cross-Validation 

Libraries: 

    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error
    from sklearn.model_selection import cross_val_score

Train/Test Split (80/20):

    X = df[best_feat]
    y = df['price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
    linreg = LinearRegression()
    linreg.fit(X_train, y_train)
    y_hat_train = linreg.predict(X_train)
    y_hat_test = linreg.predict(X_test)
    train_residuals = y_hat_train - y_train
    test_residuals = y_hat_test - y_test
    train_mse = mean_squared_error(y_train, y_hat_train)
    test_mse = mean_squared_error(y_test, y_hat_test)
    
    Train Mean Squarred Error: 

        10762973643.994843

    Test Mean Squarred Error: 

        11709630906.923727
    
Cross Validation:

    cv_1000_results = np.mean(cross_val_score(linreg, X, y, cv=1000, scoring="neg_mean_squared_error"))

    neg_mean_squared_error:

        10978008506.124075


### Recommendations

Just by adding a full, or half, bathroom, you can increase the price of your home. You can also add additional bedrooms to increase the price of your home. Be sure to keep your bathroom to bedroom ratio at/or above, 1, because the more bathrooms than bedrooms will yeild higher home prices. 



# Maximum-Covariance-Unfolding-Regression

This is the code repository for Code Repository for the paper Maximum Covariance Unfolding Regression: A Novel Covariate-based Manifold Learning Approach for Point Cloud Data by Qian Wang and Kamran Paynabar at Georgia Tech.

## Instructions:

Put the data 'XY.mat' in the folder "data" and run the file main.py. The result plots will show in the folder "results". The data should be stored as numpy arrays following the format as mentioned in the paper. The optimal process variables learned for the nominal shape in the data 'XYtest.mat' will be printed.

## Packages Requirements:
Numpy 1.21.6  
Scipy 1.7.3  
Scikit-learn 1.0.2  
Pandas 1.3.5  
Seaborn 0.11.2  
Matplotlib 3.5.3  
Mosek (for solving SDP) 9.3.21  

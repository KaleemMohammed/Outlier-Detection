
import numpy as np
import pandas as pd


#**********************************************************************************************************************#
#********************************************* Univariate Outliers Treatment *******************************************#
#**********************************************************************************************************************#

# Create dummy DataFrame
dummydf = pd.DataFrame(columns = ["Feature1"])
dummydf.Feature1 = [20,30,33,40,20,55,30,22,45,60,200]

#=======================================================================================================================
#===================================================== Method 1 ========================================================
#=======================================================================================================================

def Oulier_Std(factor, nstd):
    """
        Description: To figure out outliers

        Args: Column name with DataFrame reference and no.of standard deviation (Sigma)

        Returns: Return list of TRUE or FALSE

    """
    print("Method 1 nstd :",nstd)
    try:
        lower_limit = factor.mean() - nstd * factor.std()
        upper_limit = factor.mean() + nstd * factor.std()

        Is_Outlier = []

        for each in factor.values:
            if (each > upper_limit) or (each < lower_limit):
                Is_Outlier.append(True)
            else:
                Is_Outlier.append(False)
    except Exception as e:
        print("Error at Oulier_Std method, Solve it by :", str(e))
    return Is_Outlier

dummydf['Is_Outlier_M1'] = Oulier_Std(dummydf['Feature1'],1)

#=======================================================================================================================
#===================================================== Method 2 ========================================================
#=======================================================================================================================

def Outlier_IQR(factor,nstd):

    """
        Description: To figure out outliers

        Args: Column name with DataFrame reference and no.of standard deviation (Sigma)

        Returns: Return list of TRUE or FALSE

    """
    print("Method 2 nstd :", nstd)
    try:
        # calculate interquartile range
        Q1, Q3 = np.percentile(factor, 25), np.percentile(factor, 75)
        IQR = Q3 - Q1

        lower_limit = Q1 - nstd * IQR
        upper_limit = Q3 + nstd * IQR

        Is_Outlier = []

        for each in factor.values:
            if (each > upper_limit) or (each < lower_limit):
                Is_Outlier.append(True)
            else:
                Is_Outlier.append(False)
    except Exception as e:
        print("Error at Outlier_IQR method, Solve it by :",str(e))
    return Is_Outlier

dummydf['Is_Outlier_M2'] = Outlier_IQR(dummydf['Feature1'],1)

print(dummydf)

#**********************************************************************************************************************#
#********************************************* Multivariate Outliers Treatment *****************************************#
#**********************************************************************************************************************#

dummydf2 = pd.DataFrame(columns = ["Feature1","Feature2","Feature3"])
dummydf2.Feature1 = [20,30,33,40,20,55,30,22,45,60,200]
dummydf2.Feature2 = [20,30,500,40,20,55,30,800,45,60,200]
dummydf2.Feature3 = [20,20,33,40,20,40,30,22,45,60,600]

#=======================================================================================================================
#================================================ Mahalanobis Distance =================================================
#=======================================================================================================================

def ML_Outlier_Detection(**kwargs):

    """
        Description: To find the outliers from Multiple columns

        Args: DataFrame(Only Numeric Columns) and no.of standard deviation (Sigma)

        Returns: Row index which has outliers

    """
    df = kwargs.get("DataFrame")
    sigma = kwargs.get("sigma")

    covariance_xyz = df.cov()
    inv_covariance_xyz = np.linalg.inv(covariance_xyz)

    X_master = []

    for tag in df.columns:
        X_master.append([x_i - df[str(tag)].mean() \
                         for x_i in df[str(tag)].values])

    X_master_T = np.transpose(X_master)

    MLobies_distance = []
    for i in range(len(X_master_T)):
        MLobies_distance.append(np.sqrt(np.dot(np.dot(np.transpose(X_master_T[i]) \
                                                      , inv_covariance_xyz), X_master_T[i])))

    threshold = np.mean(MLobies_distance) * sigma  # adjust 1.5,2,3

    ML_outliers = []
    for i in range(len(MLobies_distance)):
        if MLobies_distance[i] > threshold:
            ML_outliers.append(i)

    return np.array(ML_outliers)

kwargs = {\
            'DataFrame' : dummydf2,\
            'sigma'     : 1.5
         }


print("Row index which has outliers :",ML_Outlier_Detection(**kwargs))
# Logistic Regression
## Prediction of Coronary Heart Disease Using Logistic Regression Binary Classifier
In this task we predict the Ten Year Risk of Coronary Heart Disease using Logistic Regression. Given the low correlation coefficients in the correlation matrix of the dependent and independent variables, feature selection is performed using the Chi-squared test. Analysis and accuracy of the prediction of variable, "TenYearCHD" with a classification threshold of 0.5 results in a very low f1_score of 13.47%. However, with a classification threshold of 0.2, the f1_score increases to 38.44%. 

On the basis that age and gender could be confounder variables that result in distortion of the association between exposures and outcome in the dataset, the variable age_grp is introduced to classify each record into respective age groups and stratify each age_grp data by gender. For further analysis, the dataset is reprocessed into probability values, with the assumption that values of exposure variables could be probability values such that probability of TenYearCHD given an individual is exposed is a multiple of the probability of TenYearCHD given an individual isn't exposed, where the Pr(TenYearCHD|exposed) = Pr(TenYearCHD|unexposed) * Adjusted_Risk_Ratio. The prediction is redone using classification threshold of 0.2

**Proof of Age and Gender as Confounding Variables**: https://github.com/mshobanke/Apache-Spark-MLlib-Random-Forest-Logistic-Regression

**Presentation File**: https://docs.google.com/presentation/d/1esVlCQLHBVcXPKJK2AFHA8W4pqrv6exB/edit?usp=share_link&ouid=102783274469468293710&rtpof=true&sd=true

# SFCrimeClassification
The project uses the SF crime dataset that provides nearly 12 years of crime reports from across all of San Francisco's neighborhoods. Given a time and location, the project shall predict the category of crime that occurred. This dataset contains incidents derived from SFPD Crime Incident Reporting system. The data ranges from 01/01/2003 to 05/13/2015. The training set and test set rotate every week, meaning week 1,3,5 belong to test set, week 2,4,6,8 belong to training set.

I will utilize machine learning classification techniques like Logistic Regression, SVM, Decision trees etc. to build a model that predicts the category of the crime. The best performing algorithm will be applied on the test dataset. The predicted labels i.e the category of the crime of the test test model will be evaluated via a Kaggle submission.


classify.py: Implements KNN and base code
svm.py: Implements SVM
naivebayes.py: Implemnents Naive Bayes
dtressAndRForests.py: Implements Decision Trees and Random Forests
logistic.py: Implements Logistic Regression and the plots

Python version: 2.7
Scikit Learn: 0.18


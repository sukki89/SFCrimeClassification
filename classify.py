import pandas as pd
import numpy as np
import math
import zipfile
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing

def llfun(act, pred):
    """ Logloss function for 1/0 probability
    """
    return (-(~(act == pred)).astype(int) * math.log(1e-15)).sum() / len(act)

def readData():
    train = pd.read_csv('data/train.csv', parse_dates=['Dates'])
    test = pd.read_csv('data/test.csv', parse_dates=['Dates'])
    print("Number of cases in the training set: %s" % len(train))
    print("Number of cases in the testing set: %s" % len(test))
    return (train,test)

def applyFunction(train, inputCol, check, outputCol):
    train[outputCol] = train[inputCol].apply(lambda x:1 if x == check else 0 )
    return train 

def convertToFeatures(train):
    train = applyFunction(train, 'DayOfWeek',"Sunday", "sun")
    train = applyFunction(train, 'DayOfWeek',"Monday", "mon")
    train = applyFunction(train, 'DayOfWeek', "Tuesday", "tues")
    train = applyFunction(train, 'DayOfWeek',"Wednesday", "wed")
    train = applyFunction(train, 'DayOfWeek',"Thursday", "thur")
    train = applyFunction(train, 'DayOfWeek',"Friday", "fri")
    train = applyFunction(train, 'DayOfWeek',"Saturday", "sat")
    train = applyFunction(train, 'PdDistrict',"BAYVIEW", "BAYVIEW")
    train = applyFunction(train, 'PdDistrict',"CENTRAL", "CENTRAL")
    train = applyFunction(train, 'PdDistrict',"INGLESIDE", "INGLESIDE")
    train = applyFunction(train, 'PdDistrict',"MISSION", "MISSION")
    train = applyFunction(train, 'PdDistrict',"NORTHERN", "NORTHERN")
    train = applyFunction(train, 'PdDistrict',"PARK", "PARK")
    train = applyFunction(train, 'PdDistrict',"RICHMOND", "RICHMOND")
    train = applyFunction(train, 'PdDistrict',"SOUTHERN", "SOUTHERN")
    train = applyFunction(train, 'PdDistrict',"TARAVAL", "TARAVAL")
    train = applyFunction(train, 'PdDistrict',"TENDERLOIN", "TENDERLOIN")
    return train 

def divideIntoTrainAndEvaluationSet(fraction, train):
    msk = np.random.rand(len(train)) < fraction
    trainOnly = train[msk]
    evaluateOnly = train[~msk]
    print("Number of cases in the training only set: %s" % len(trainOnly))
    print("Number of cases in the evaluation  set: %s" % len(evaluateOnly))
    return(trainOnly,evaluateOnly)

def classify(name, train, evaluate, test):
    if(name == "knn"):
        return knnClassifier(train, evaluate, test)
    elif(name =="svm"):
        return svmClassifier(train, evaluate, test)
    elif(name == "logit"):
        return logisticRegressionClassifier(train, evaluate, test)
    elif(name == "dtrees"):
        return dtreesClassifier(train, evaluate, test)
    else:
        print(" Specify the right name of the classifier : knn/svm/logit/dtrees")


def knnClassifier(train, evaluate, test):
    print('In K nerarest neighbour')
    x = train[['X', 'Y']]
    y = train['Category'].astype('category')
    actual = evaluate['Category'].astype('category')

    # Fit
    logloss = []
    for i in range(1, 50, 1):
        knn = KNeighborsClassifier(n_neighbors=i)
        knn.fit(x, y)
    
        # Predict on test set
        outcome = knn.predict(evaluate[['X', 'Y']])
    
        # Logloss
        logloss.append(llfun(actual, outcome))

    plt.plot(logloss)
    plt.savefig('n_neighbors_vs_logloss.png')
   
    # Fit test data
    x_test = test[['X', 'Y']]
    knn = KNeighborsClassifier(n_neighbors=40)
    knn.fit(x, y)
    outcomes = knn.predict(x_test)
    #return outcomes 
   
    # Move this to a separate function later

    submit = pd.DataFrame({'Id': test.Id.tolist()})
    for category in y.cat.categories:
        submit[category] = np.where(outcomes == category, 1, 0)
    
    submit.to_csv('k_nearest_neigbour.csv', index = False)

    
def svmClassifier(train, test):
    print('In SVM')

def logisticRegressionClassifier(train, test):
    print('In Logistic Regression')

def dtreesClassifier(train, test):
    print('In decision trees')

#def createSubmissionFile(lables, fileName):
	
def main():
   (train, test) = readData()
   print train.columns.values
   train = convertToFeatures(train)
   test = convertToFeatures(test)
   print train.columns.values
   print train[1115:1130]
   (trainOnly,evaluateOnly) = divideIntoTrainAndEvaluationSet(0.8, train)
   

   # Call the classifiers - replace with your classifier
   #predictedLabels = classify("knn",trainOnly, evaluateOnly, test)
   #print(predictedLabels)
   
main()

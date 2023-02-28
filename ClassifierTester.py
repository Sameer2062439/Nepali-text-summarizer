class ClassifierTester:
    import numpy as np
    import pandas as pd
    import pickle    
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.feature_extraction.text import CountVectorizer

    from PreProcessor import PreProcessor
    # from RFclassifier import RFclassifier
    # from SVMClassifier import SVMClassifier


    # def __init__(self, path):        
    #     self.path = path

    def removeComma(convertedDesignTolist):

        newList = []

        for temp in convertedDesignTolist:
            strings = [str(j) for j in temp]
            a_string = "".join(strings)
            an_integer = int(a_string)
            newList.append(an_integer)
        
        return newList
    
    dataset = pd.read_csv('../data/test.csv')

    predictors = dataset[['designation', 'experience']]

    exp = predictors['experience'].values.tolist()

    vecto = CountVectorizer()
    convertedDesign = vecto.fit_transform(predictors['designation'])
    convertedDesignToarray = convertedDesign.toarray()
    convertedDesignTolist = convertedDesignToarray.tolist()
    newList = removeComma(convertedDesignTolist)

    # print(newList)

    diction = {'designation':newList, 'experience':exp} #StandardScaler doesn't help to increase model performance here
    df = pd.DataFrame(diction)
    # print(df)

    predArray = np.array(df)

    # # load the model from disk
    # #To test for Random Forest
    # filename = 'Models/RFClassifier.sav'
    # loaded_model = pickle.load(open(filename, 'rb'))

    # load the model from disk
    #To test for Linear SVM or RandomForest
    filename = 'Models/SVMClassifier.sav'
    loaded_model = pickle.load(open(filename, 'rb'))

    scaler = StandardScaler()
    X_test = scaler.fit_transform(predArray)
    y_pred = loaded_model.predict(X_test)
    # print(y_pred)

    # RFclassifier = RFclassifier()
    # X_test = RFclassifier.scaler.transform(predArray)

    # y_pred = RFclassifier.classifier.predict(X_test)
    # print(y_pred)

    dataProcessor = PreProcessor()
    reversedWordspred = dataProcessor.encoder.inverse_transform(y_pred)
    print(reversedWordspred)
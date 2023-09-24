import pandas as pd
from pycaret.classification import ClassificationExperiment

def load_data(filepath):
    """
    Loads churndata into a DataFrame from a string filepath.
    """
    dfn = pd.read_csv(filepath, index_col='customerID')
    dfn['PhoneService'] = dfn['PhoneService'].replace({'No': 0, 'Yes': 1})
    dfn['Contract'] = dfn['Contract'].replace({'Month-to-month': 0, 'One year': 1, 'Two year':2})
    dfn['PaymentMethod'] = dfn['PaymentMethod'].replace({'Electronic check': 0, 'Mailed check': 1, 'Bank transfer (automatic)':2, 'Credit card (automatic)':3})
    return dfn


def make_predictions(dfn):
    """
    Uses the pycaret best model to make predictions on data in the df dataframe.
    """
    classifier = ClassificationExperiment()
    model = classifier.load_model('pycaret_model')
    predictions = classifier.predict_model(model, data=dfn)
    #predictions.rename({'Label': 'Churn_prediction'}, axis=1, inplace=True)
    #predictions['Churn_prediction'].replace({1: 'Churn', 0: 'No churn'},inplace=True)
    return predictions#['Churn_prediction']


if __name__ == "__main__":
    dfn = load_data('~/Documents/MSDS_VaishaliWork/MSDS600_DataScience/Week5/Assignment/new_churn_data_unmodified.csv')
    predictions = make_predictions(dfn)
    print('predictions:')
    print(predictions)

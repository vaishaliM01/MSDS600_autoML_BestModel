import pandas as pd
from pycaret.classification import ClassificationExperiment

def load_data(filepath):
    """
    Loads churndata into a DataFrame from a string filepath.
    """
    df = pd.read_csv(filepath, index_col= 'customerID')
    return df


def make_predictions(df):
    """
    Uses the pycaret best model to make predictions on data in the df dataframe.
    """
    classifier = ClassificationExperiment()
    model = classifier.load_model('pycaret_model')
    predictions = classifier.predict_model(model, data=df)
    #predictions.rename({'Label': 'Churn_prediction'}, axis=1, inplace=True)
    #predictions['Churn_prediction'].replace({1: 'Churn', 0: 'No churn'},inplace=True)
    return predictions#['Churn_prediction']


if __name__ == "__main__":
    df = load_data('~/Documents/MSDS_VaishaliWork/MSDS600_DataScience/Week5/Assignment/new_churn_data.csv')
    predictions = make_predictions(df)
    print('predictions:')
    print(predictions)

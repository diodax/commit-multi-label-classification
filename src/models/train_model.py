# -*- coding: utf-8 -*-
import click
import logging
import pandas as pd

from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix
from pandas import DataFrame
# noinspection PyUnresolvedReferences
from skmultilearn_classifier import ScikitMultilearnClassifier


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
def main(input_filepath):
    """
    Starting point of the project. Receives the location of the dataset as a
    command-line Path argument.
    """
    logger = logging.getLogger(__name__)
    logger.info('Making final TD-IDF scores set from processed data')

    # Get the data/processed/refactoring-commits-sample.csv file
    dataset = pd.read_csv(input_filepath, index_col=0)
    logger.info('Loaded data file ' + input_filepath + ' with ' + str(len(dataset)) + ' rows')

    # Drop the columns that aren't needed to build the TF-IDF scores
    dataset = dataset.drop(['Id', 'RefactoringId', 'CommitId', 'AppName'], axis=1)
    dataset = dataset.dropna()

    # x = dataset.iloc[:, 0:1].values
    y = dataset.iloc[:, 1:-1].values

    x1 = dataset['Message'].values.astype(str)

    vec = TfidfVectorizer(min_df=5, max_df=0.7).fit(x1)
    x = DataFrame(vec.transform(x1).todense(), columns=vec.get_feature_names())
    x_train, x_test, y_train, y_test = train_test_split(x, y)

    # -----------------------------------------------------

    # Split into training and test sets
    # train, test = train_test_split(dataset, test_size=0.3, shuffle=True)
    # x_train = train.Message
    # x_test = test.Message

    # categories = ['IsFunctional', 'IsNonFunctional', 'IsTesting', 'IsBugFix', 'IsStructuralDesignOptimization',
    #               'IsCodeOptimization', 'IsAdministrativeManagement', 'IsUnknown']

    # y_train = csr_matrix(train[categories].values)
    # y_test = csr_matrix(test[categories].values)
    #
    # vec = TfidfVectorizer(min_df=5, max_df=0.7)
    # vec.fit(dataset['Message'].values.astype(str))
    # x_train = vec.transform(x_train)
    # x_test = vec.transform(x_test)

    # Option 1: Using the Scikit-multilearn library for multi-label classification
    skmultilearn = ScikitMultilearnClassifier(x_train, y_train, x_test, y_test)

    a, f = skmultilearn.apply_binary_relevance()
    print("Binary Relevance Approach - Acc: " + str(a) + ", F1 Score: " + str(f))
    a, f = skmultilearn.apply_classifier_chain()
    print("Classifier Chain Approach - Acc: " + str(a) + ", F1 Score: " + str(f))
    a, f = skmultilearn.apply_label_powerset()
    print("Label Powerset Approach - Acc: " + str(a) + ", F1 Score: " + str(f))
    a, f = skmultilearn.apply_mlknn()
    print("MLkNN Approach - Acc: " + str(a) + ", F1 Score: " + str(f))


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()

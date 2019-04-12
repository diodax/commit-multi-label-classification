# -*- coding: utf-8 -*-
import click
import logging
import pandas as pd

from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """
    Starting point of the project. Receives the location of the dataset as a
    command-line Path argument.
    """
    logger = logging.getLogger(__name__)
    logger.info('Making final TD-IDF scores set from processed data')

    # Get the data/processed/refactoring-commits-sample.csv file
    dataset = pd.read_csv(input_filepath)
    logger.info('Loaded data file ' + input_filepath + ' with ' + str(len(dataset)) + ' rows')

    # Drop the columns that aren't needed to build the TF-IDF scores
    dataset = dataset.drop(['Id', 'RefactoringId', 'CommitId', 'AppName'], axis=1)
    dataset = dataset.dropna()

    vec = TfidfVectorizer(min_df=5, max_df=0.7)
    vec.fit(dataset['Message'].values.astype(str))
    tfidf_scores = vec.transform(dataset.Message)
    df_scores = pd.DataFrame(tfidf_scores.toarray())

    # Generate new DataFrame with the TF-IDF scores
    logger.info('Saving TF-IDF scores in a new .pkl file...')

    # Save the TF-IDF scores on data models/tf-idf-scores.csv
    logger.info('Saved processed scores on ' + output_filepath)
    df_scores.to_pickle(output_filepath)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()

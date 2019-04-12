# -*- coding: utf-8 -*-
import click
import logging
import string
import json
import os
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import pandas as pd
import nltk
import contractions
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.corpus import stopwords


# Download NLTK resources
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

# String constants
CUSTOM_STOPWORDS_FILENAME = 'stopwords_custom.json'


def get_part_of_speech(term):
    if term == "":
        return ""
    temp = nltk.pos_tag([term.lower()])
    if len(temp) >= 1:
        return temp[0][1]
    return ""


def get_stopwords(filename):
    """
    Given a .JSON file location with a list of stopwords, extract it and return them as an array
    """
    # Open our JSON file and load it into python
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)) as json_data:
        data = json.load(json_data)
    return data


def preprocess(text):
    """
    Applies the following pre-processing steps to a given string:
    Step 1: Convert to lower case, replace slashes (/) with spaces
    Step 1: Tokenize the strings
    Step 3: Retain only nouns, verbs, adjectives and adverbs (using a POS-tagger)
    Step 4: Remove default English stopwords
    Step 5: Remove custom domain stopwords
    Step 6: Lemmatize each word
    """
    # convert to lowercase
    text = text.lower()

    # remove whitespaces from start and end of text
    text = text.strip()

    # expand contractions
    text = contractions.fix(text)

    # remove urls
    text = re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', text)

    # remove alphanumeric words
    text = re.sub(r'\w*\d\w*', '', text)

    # remove numbers
    text = re.sub(r'\d+', '', text)

    # remove special characters
    text = re.sub(r'/[.,\/#!$%\^&\*;:{}=\-_`~()]/g', '', text)

    # tokenize the words
    tokens = nltk.word_tokenize(text)

    # remove NLTK stopwords
    stop_words = stopwords.words('english')
    tokens = [i for i in tokens if i not in stop_words]

    # remove punctuations
    tokens = [i for i in tokens if i not in string.punctuation]

    # retain  verbs, adjectives, and adverbs
    pos_retain = ['NN', 'NNS', 'NNP', 'NNPS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS']
    # pos_retain = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS']
    tokens = [i for i in tokens if get_part_of_speech(i) in pos_retain]

    # reduce each word to its lemma
    wordnet_lemmatizer = WordNetLemmatizer()
    tokens = [wordnet_lemmatizer.lemmatize(x) for x in tokens]

    # Remove custom stop words
    custom_stop_words = get_stopwords(CUSTOM_STOPWORDS_FILENAME)
    # tokens = [i for i in tokens if i not in custom_stop_words]
    tokens = [i for i in tokens if not any(substring in i for substring in custom_stop_words)]

    text = ' '.join(tokens)
    return text


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_interim_filepath', type=click.Path())
@click.argument('output_final_filepath', type=click.Path())
def main(input_filepath, output_interim_filepath, output_final_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    # Get the data/raw/refactoring-commits-sample.csv file
    # Also excludes the last column from each row due to extra commas on the csv
    dataset_raw = pd.read_csv(input_filepath)
    logger.info('Loaded data file ' + input_filepath + ' with ' + str(len(dataset_raw)) + ' rows')

    columns = ['Id', 'RefactoringId', 'CommitId', 'AppName', 'Message', 'IsFunctional', 'IsNonFunctional', 'IsTesting',
               'IsBugFix', 'IsStructuralDesignOptimization', 'IsCodeOptimization', 'IsAdministrativeManagement',
               'IsUnknown']
    data = []

    for index, row in dataset_raw.iterrows():
        if not row['Needs Review']:
            data_row = [row['Id'], row['RefactoringId'], row['CommitId'], row['app']]
            message_row = row['Message'].replace('\n', '').replace('\r', '')
            data_row.append(message_row)

            labels = row['Label'].split('; ')

            # Based on the strings present on the Labels column, fill the rest of the binary columns
            data_row.append(True if 'Functional' in labels else False)
            data_row.append(True if 'Non-Functional' in labels else False)
            data_row.append(True if 'Testing' in labels else False)
            data_row.append(True if 'Bug Fix' in labels else False)
            data_row.append(True if 'Structural Design Optimization' in labels else False)
            data_row.append(True if 'Code Optimization' in labels else False)
            data_row.append(True if 'Administrative/Management' in labels else False)
            data_row.append(True if 'Unknown' in labels else False)

            data.append(data_row)
    dataset_normalized = pd.DataFrame(data=data, columns=columns)
    # Save the normalized subset on data data/interim/refactoring-commits-normalized.csv
    logger.info(
        'Saved normalized table on ' + output_interim_filepath + ' with ' + str(len(dataset_normalized)) + ' rows')
    dataset_normalized.to_csv(output_interim_filepath, encoding='utf-8', index=False)

    # Start the actual pre-processing steps in the Message column
    logger.info('Applying pre-processing steps on the "Message" column...')
    dataset_normalized['Message'] = dataset_normalized['Message'].apply(preprocess)

    # Save the processed subset on data data/processed/refactoring-commits-processed.csv
    logger.info(
        'Saved processed results on ' + output_final_filepath + ' with ' + str(len(dataset_normalized)) + ' rows')
    dataset_normalized.to_csv(output_final_filepath, encoding='utf-8')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()

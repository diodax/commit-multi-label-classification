# -*- coding: utf-8 -*-
import click
import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from wordcloud import WordCloud

# String constants
CATEGORY_FREQ_PLOT_FILENAME = 'category-frequency-barplot.png'
LABEL_COUNT_PLOT_FILENAME = 'multilabel-count-barplot.png'
WORD_CLOUD_FILENAME = 'word-cloud.png'


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_folder', type=click.Path())
def main(input_filepath, output_folder):
    """
    Starting point of the project. Receives the location of the dataset as a
    command-line Path argument.
    """
    logger = logging.getLogger(__name__)

    # Get the data/processed/refactoring-commits-sample.csv file
    dataset = pd.read_csv(input_filepath, index_col=0)
    logger.info('Loaded data file ' + input_filepath + ' with ' + str(len(dataset)) + ' rows')
    dataset = dataset.dropna()

    # Generating the Category Frequency Bar Plot

    categories = list(dataset.columns.values)[-8:]
    plt.figure(figsize=(15, 8))
    ax1 = sns.barplot(categories, dataset.iloc[:, 5:].sum().values)

    plt.title("Comments in each category", fontsize=24)
    plt.ylabel('Number of comments', fontsize=18)
    plt.xlabel('Comment Type ', fontsize=18)

    # the plot gets saved to 'reports/figures/category-frequency-barplot.png'
    plt.savefig(output_folder + CATEGORY_FREQ_PLOT_FILENAME)
    logger.info('Created plot graph at ' + output_folder + CATEGORY_FREQ_PLOT_FILENAME)

    # Generating the Multi-Label Count Bar Plot

    row_sums = dataset.iloc[:, 5:].astype(bool).sum(axis=1)
    multi_label_counts = row_sums.value_counts()
    multi_label_counts = multi_label_counts.iloc[1:]

    sns.set(font_scale=2)
    plt.figure(figsize=(15, 8))

    ax2 = sns.barplot(multi_label_counts.index, multi_label_counts.values)

    plt.title("Comments having multiple labels ")
    plt.ylabel('Number of comments', fontsize=18)
    plt.xlabel('Number of labels', fontsize=18)

    # the plot gets saved to 'reports/figures/multilabel-count-barplot.png'
    plt.savefig(output_folder + LABEL_COUNT_PLOT_FILENAME)
    logger.info('Created plot graph at ' + output_folder + LABEL_COUNT_PLOT_FILENAME)

    plt.figure(figsize=(40, 25))
    text = dataset.Message.values
    cloud = WordCloud(
        background_color='black',
        collocations=False,
        width=2500,
        height=1800
    ).generate(" ".join(text))
    plt.axis('off')
    plt.margins(x=0, y=0)
    plt.title("Processed Messages", fontsize=40)
    plt.imshow(cloud)

    # the plot gets saved to 'reports/figures/word-cloud.png'
    plt.savefig(output_folder + WORD_CLOUD_FILENAME)
    logger.info('Created word cloud at ' + output_folder + WORD_CLOUD_FILENAME)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()

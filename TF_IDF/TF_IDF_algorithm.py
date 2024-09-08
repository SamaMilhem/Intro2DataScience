import pandas as pd
import numpy as np
import re

TF_IDF_COLUMN = 'Description'
TITLE = 'Music Festival'

def tfidf(df, terms):
    """
        Calculate the Term Frequency-Inverse Document Frequency (TF-IDF) values for a set of terms in a given DataFrame.

        Parameters:
        ----------
        df : pandas.DataFrame
            The input DataFrame where each row corresponds to a document, and the specified column contains
             the text of the documents.

        terms : list of str
            A list of terms for which the TF-IDF scores will be calculated.

        Returns:
        -------
        pandas.DataFrame
            A DataFrame containing the TF-IDF scores for each document (rows) and each term (columns).
        """

    tf = pd.DataFrame(index=df.index, columns=list(terms)) # tf calculation
    for term in terms:
        tf[term] = df[('%s' % TF_IDF_COLUMN)].apply(lambda x: len(re.findall(r'\b{}\b'.format(re.escape(term)),
                                                                             x.lower())))
    df_counts = tf.astype(bool).sum(axis=0)  # DF calculation
    idf = np.log10(len(df) / df_counts) # idf calculation
    return tf.multiply(idf, axis=1) # tf-idf

def analyze_music_festival(terms, output_file, input_file):

    df = pd.read_csv(input_file)
    tfidf_scores = tfidf(df, terms)
    cols = [TITLE] + [col for col in tfidf_scores]
    tfidf_scores[TITLE] = df[TITLE]
    tfidf_scores = tfidf_scores[cols]
    output_path = './Output/' + output_file + '.csv'
    tfidf_scores.to_csv(output_path, sep=',', encoding='utf-8', index=False)

if __name__ == '__main__':
    terms = ["annual", "music", "festival", "soul", "jazz", "belgium", "hungary",
             "israel", "rock", "dance", "desert", "electronic", "arts"]
    output_file_path = 'tfidf_scores'
    input_file_path = 'music_festivals.csv'
    analyze_music_festival(terms, output_file_path, input_file_path)
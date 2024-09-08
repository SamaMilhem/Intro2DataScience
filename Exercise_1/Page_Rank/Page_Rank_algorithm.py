import json
import re
import string
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
import Web_Crawling

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TF_IDF.TF_IDF_algorithm import *


def get_sentences (data):
    """
    Parameters:
    ----------
    data : str
        The input string containing raw text.

    Returns:
    -------
    list of str
        A list of sentences after cleaning and splitting the text.
        Each sentence is split based on periods (".")."""

    sentences = re.sub(r'\[.*?]| [A-Z]\.', '', data).lower()
    sentences = re.sub(r"(?<!\w)['\"](?!\w)", '', sentences)
    sentences = re.sub(r'\b\d+(\.\d+)?\b', '', sentences)
    sentences = ' '.join(sentences.split())
    return sentences.split(".")
def process_json_file(file_name):
    """
       Processes a JSON file containing Wikipedia page text and returns two lists:
       a list of cleaned sentences and a list of original sentences.

       Parameters:
       ----------
       file_name : str
           The name of the JSON file (without the extension) that contains the text data.
           The file is expected to be located in the '../Output_WikiPages/' directory.

       Returns:
       -------
       tuple of lists
           - cleaned_sentences: A list of sentences that have been tokenized, cleaned, and filtered to remove punctuation, numbers, stopwords, and non-alphabetic characters.
           - original: A list of the original sentences from the text before cleaning.
    """

    with open('./Output_WikiPages/' + file_name + '.json', 'r', encoding='utf-8') as file:
        data = json.load(file)[file_name]
        sentences = get_sentences(data)
    # Prepare for filtering
    stop_words = set(stopwords.words('english'))
    table = str.maketrans('', '', string.punctuation + string.digits)

    # Process each sentence
    cleaned_sentences = []
    original = []
    for sentence in sentences:
        # Tokenize sentence into words
        tokens = word_tokenize(sentence)
        # Remove punctuation, numbers, and stopwords
        filtered_tokens = [word.translate(table) for word in tokens if
                           word.isalpha() and word.lower() not in stop_words and word.isascii()]
        cleaned_sentence = ' '.join(filtered_tokens)
        if cleaned_sentence != "":
            sentence = sentence.replace('\n', '')
            original.append(sentence)
            cleaned_sentences.append(cleaned_sentence)

    return cleaned_sentences,original

def calculate_tf(sentences):
    """
    Calculate the term frequency (TF) for each word in each sentence.

    Parameters:
    ----------
    sentences : list of str
        List of cleaned sentences.

    Returns:
    -------
    list of dict
        A list of dictionaries where each dictionary represents the TF of words in a sentence.
    """
    tf_list = []
    for sentence in sentences:
        words = sentence.split()
        word_count = Counter(words)
        sentence_length = len(words)
        tf = {word: count / sentence_length for word, count in word_count.items()}
        tf_list.append(tf)
    return tf_list


def calculate_sentence_count(sentences):
    """
    Calculate the number of sentences in which each word appears.

    Parameters:
    ----------
    sentences : list of str
        List of cleaned sentences.

    Returns:
    -------
    dict
        A dictionary where keys are words and values are the count of sentences containing the word.
    """
    sentence_count = {}
    for sentence in sentences:
        words = set(sentence.split())
        for word in words:
            if word in sentence_count:
                sentence_count[word] += 1
            else:
                sentence_count[word] = 1
    return sentence_count


def calculate_idf(sentence_count, total_sentences):
    """
    Calculate the inverse document frequency (IDF) for each word.

    Parameters:
    ----------
    sentence_count : dict
        Dictionary where keys are words and values are the count of sentences containing the word.
    total_sentences : int
        Total number of sentences.

    Returns:
    -------
    dict
        A dictionary where keys are words and values are their IDF scores.
    """
    idf_words = {}
    for word, count in sentence_count.items():
        idf_words[word] = np.log10(total_sentences / count)
    return idf_words


def build_weights_graph(sentences, tf_list, idf_words):
    """
    Build a weights graph using the cosine similarity between sentences.

    Parameters:
    ----------
    sentences : list of str
        List of cleaned sentences.
    tf_list : list of dict
        List of TF dictionaries for each sentence.
    idf_words : dict
        Dictionary of IDF scores for each word.

    Returns:
    -------
    numpy.ndarray
        A 2D array representing the weights graph between sentences.
    """
    n = len(sentences)
    weights_graph = np.zeros((n, n))

    for i in range(n):
        words_i = set(sentences[i].split())
        sum_words_i = sum(((tf_list[i][word] * idf_words[word]) ** 2) for word in words_i)
        d1 = np.sqrt(sum_words_i)

        for j in range(n):
            words_j = set(sentences[j].split())
            sum_words_j = sum(((tf_list[j][word] * idf_words[word]) ** 2) for word in words_j)
            d2 = np.sqrt(sum_words_j)

            sum_of_tf = sum(((tf_list[i][word] * tf_list[j][word]) * (idf_words[word]) ** 2)
                            for word in words_i if word in words_j)
            if d1 != 0 and d2 != 0:
                weights_graph[i, j] = sum_of_tf / (d1 * d2)

    # Normalize weights graph
    weights_graph /= np.where(np.sum(weights_graph, axis=0) == 0, 1, np.sum(weights_graph, axis=0))
    return weights_graph


def calculate_pagerank(weights_graph, damping_factor=0.85, tolerance=0.000005):
    """
    Calculate the PageRank scores for each sentence using the weights graph.

    Parameters:
    ----------
    weights_graph : numpy.ndarray
        The weights graph where each entry represents the similarity between sentences.
    damping_factor : float, optional
        The damping factor for the PageRank algorithm, by default 0.85.
    tolerance : float, optional
        The convergence tolerance, by default 0.000005.

    Returns:
    -------
    numpy.ndarray
        A 1D array representing the PageRank scores of the sentences.
    """
    n = len(weights_graph)
    scores_old = np.ones(n) / n

    while True:
        scores_new = (1 - damping_factor) / n + damping_factor * np.dot(weights_graph, scores_old)
        if np.linalg.norm(scores_new - scores_old, 1) < tolerance:
            break
        scores_old = scores_new

    return scores_new


def textsum():
    sentences_dict = {}
    for fruit, _ in Web_Crawling.FRUITS_LINKS.items():
        sentences_dict[fruit], original = process_json_file(fruit)

        # Calculate term frequency (TF)
        tf_list = calculate_tf(sentences_dict[fruit])

        # Calculate sentence count for each word
        sentence_count = calculate_sentence_count(sentences_dict[fruit])

        # Calculate inverse document frequency (IDF)
        idf_words = calculate_idf(sentence_count, len(sentences_dict[fruit]))

        # Build the weights graph
        weights_graph = build_weights_graph(sentences_dict[fruit], tf_list, idf_words)

        # Calculate PageRank scores
        pagerank_scores = calculate_pagerank(weights_graph)

        # Rank sentences based on PageRank scores
        ranked_sentences = [original[idx] for idx in np.argsort(pagerank_scores)]
        sentences_dict[fruit] = ranked_sentences[-5:]  # Select top 5 sentences

    return sentences_dict


def analyze_words(summary):
    """
    Analyze and extract distinct words from a given summary, removing stopwords, punctuation, and non-alphabetic tokens.

    Parameters:
    ----------
    summary : list of str
        A list of sentences (summary) to process.

    Returns:
    -------
    set of str
        A set of distinct filtered words from the summary.
    """
    words = set()
    stop_words = set(stopwords.words('english'))
    table = str.maketrans('', '', string.punctuation + string.digits)

    for sentence in summary:
        # Tokenize sentence into words
        tokens = word_tokenize(sentence)
        # Filter out punctuation, numbers, stopwords, and non-ASCII characters
        filtered_tokens = [word.translate(table) for word in tokens if
                           word.isalpha() and word.lower() not in stop_words and word.isascii()]
        words.update(filtered_tokens)

    return words


def create_tf_idf_dataframe(summaries, distinct_words):
    """
    Create a DataFrame containing TF-IDF scores for distinct words across fruit descriptions.

    Parameters:
    ----------
    summaries : list of list
        A list of [fruit, summary] pairs, where summary is a string of combined sentences.

    distinct_words : set of str
        A set of distinct words to calculate TF-IDF for.

    Returns:
    -------
    pd.DataFrame
        A DataFrame containing TF-IDF scores.
    """
    df = pd.DataFrame(summaries, columns=['Fruit', 'Description'])
    return tfidf(df, distinct_words)


def get_top_words_from_tfidf(tf_idf):
    """
    Extract the top 3 words with the highest TF-IDF values from each row in the DataFrame.

    Parameters:
    ----------
    tf_idf : pd.DataFrame
        A DataFrame containing the TF-IDF scores for distinct words.

    Returns:
    -------
    set
        A set of words that have the highest TF-IDF values across all rows.
    """
    top_words = set()

    for index, row in tf_idf.iterrows():
        largest_values_cols = row.nlargest(3).index
        top_words.update(largest_values_cols)

    return top_words


def save_combined_dataframe(tf_idf, df, top_words):
    """
    Save the combined DataFrame with top words and their TF-IDF scores to a CSV file.

    Parameters:
    ----------
    tf_idf : pd.DataFrame
        The TF-IDF DataFrame.

    df : pd.DataFrame
        The original DataFrame with fruit descriptions.

    top_words : set
        The set of top words to include in the final output.

    Returns:
    -------
    None
    """
    combined_dataframe = pd.concat([df['Fruit'], tf_idf[list(top_words)]], axis=1)
    combined_dataframe.to_csv("./Output_files/combined_fruits.csv", index=False)


def top_words():
    """
    Generate a summary for each fruit, analyze top words, and save the results to a CSV file.

    Returns:
    -------
    None
    """
    sentences_dict = textsum()
    distinct_words = set()
    summaries = []

    # Analyze words and build summaries
    for fruit, summary_sentences in sentences_dict.items():
        distinct_words.update(analyze_words(summary_sentences))
        summary = " ".join(summary_sentences)
        summaries.append([fruit, summary])

    # Create TF-IDF DataFrame
    tf_idf = create_tf_idf_dataframe(summaries, distinct_words)

    # Extract top words from TF-IDF
    top_words_set = get_top_words_from_tfidf(tf_idf)

    # Save combined data to CSV
    save_combined_dataframe(tf_idf, pd.DataFrame(summaries, columns=['Fruit', 'Description']), top_words_set)

if __name__ == '__main__':
    top_words()
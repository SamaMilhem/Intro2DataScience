import re
import nltk
import numpy as np
from bs4 import BeautifulSoup
from nltk import TweetTokenizer, PorterStemmer
from nltk.corpus import stopwords
from nltk.probability import FreqDist
import requests
import matplotlib.pyplot as plt
from wordcloud import WordCloud

nltk.download('averaged_perceptron_tagger')

# Fetching a list of additional stopwords from an external source
stopwords_list = requests.get(
    "https://gist.githubusercontent.com/rg089/35e00abf8941d72d419224cfd5b5925d/raw/12d899b70156fd0041fa9778d657330b024b959c/stopwords.txt").content
stopwords = set(stopwords_list.decode().splitlines()).union(set(stopwords.words('english')))


def text_tokenize(paragraphs):
    """
    Tokenizes the input paragraphs into individual words, processes them by removing punctuation and splitting hyphenated words.

    Parameters:
    ----------
    paragraphs : list of str
        List of paragraphs (text) to tokenize.

    Returns:
    -------
    tokens : list of str
        List of tokenized words, with punctuations removed and hyphenated words split.
    """
    tokens = []
    for paragraph in paragraphs:
        paragraph = paragraph.lower().replace("’", "")  # lower case + adjust words with apostrophes like don't
        tokenizer = TweetTokenizer()
        tokens_addition = [word for word in tokenizer.tokenize(paragraph)]
        for token in tokens_addition:
            if '-' in token:
                words = token.split('-')
                for word in words:
                    tokens_addition.append(word)
                tokens_addition.remove(token)
        tokens_addition = [word for word in tokens_addition if word.isalpha()]  # removing punctuations
        tokens.extend(tokens_addition)
    return tokens


def counting_terms(tokens, title, save_path=None):
    """
    Plots the log rank vs log frequency distribution of terms (Zipf's Law) in the token list, and saves the plot if a path is provided.

    Parameters:
    ----------
    tokens : list of str
        List of tokenized words.
    title : str
        Title for the plot.
    save_path : str, optional
        Path to save the figure. If None, the figure will not be saved.

    Returns:
    -------
    sorted_tokens : list of tuples
        List of sorted tokens by their frequency in descending order, in the format (word, frequency).
    """
    frequency_distribution = FreqDist(tokens)
    sorted_tokens = sorted(frequency_distribution.items(), key=lambda token: token[1], reverse=True)
    plt.figure(figsize=(10, 6))
    plt.plot(np.log10(range(1, len(sorted_tokens) + 1)), np.log10([frequency for _, frequency in sorted_tokens]),
             marker='o', linestyle='-', color='blue')
    plt.title(title)
    plt.xlabel('Log Rank')
    plt.ylabel('Log Frequency')
    plt.grid(True)

    if save_path:
        plt.savefig(save_path, format='png', dpi=300)  # Save the figure as PNG
        print(f"Figure saved to {save_path}")

    plt.show()
    return sorted_tokens


def list_of_N_top_tokens(N, sorted_tokens):
    """
    Extracts the top N tokens from the sorted token list.

    Parameters:
    ----------
    N : int
        Number of top tokens to extract.
    sorted_tokens : list of tuples
        Sorted list of tokens (word, frequency).

    Returns:
    -------
    top_tokens : list of str
        List of the top N tokens.
    """
    counter = 0
    top_tokens = []
    for token in sorted_tokens:
        top_tokens.append(token[0])
        counter += 1
        if counter == N:
            break
    return top_tokens


def remove_stopwords(paragraphs):
    """
    Tokenizes the input paragraphs and removes stopwords.

    Parameters:
    ----------
    paragraphs : list of str
        List of paragraphs (text) to process and remove stopwords from.

    Returns:
    -------
    all_tokens : list of str
        List of tokenized words with stopwords removed.
    """
    all_tokens = []
    for paragraph in paragraphs:
        tokenizer = TweetTokenizer()
        text = paragraph.lower()
        tokens = [word for word in tokenizer.tokenize(text) if word not in stopwords]
        for token in tokens:
            if '-' in token:
                words = token.split('-')
                for word in words:
                    if word not in stopwords:
                        tokens.append(word)
                tokens.remove(token)
        tokens = [word for word in tokens if word.isalpha()]  # removing punctuations
        all_tokens.extend(tokens)
    return all_tokens


def stemming_tokens(tokens):
    """
    Stems the tokens using the Porter Stemmer.

    Parameters:
    ----------
    tokens : list of str
        List of tokenized words to be stemmed.

    Returns:
    -------
    stemmed_words : list of str
        List of stemmed words.
    """
    stemmer = PorterStemmer()
    stemmed_words = [stemmer.stem(word) for word in tokens]
    return stemmed_words


def pos_tag_print(tagged):
    """
    Prints the POS tags for each word in a tagged list.

    Parameters:
    ----------
    tagged : list of tuples
        List of tuples where each tuple is (word, POS tag).

    Returns:
    -------
    None
    """
    for word, tag in tagged:
        print(f"{word}: {tag}\n")


def pos_tag(tokens):
    """
    Performs part-of-speech (POS) tagging on the tokenized text.

    Parameters:
    ----------
    tokens : list of str
        List of tokenized words to be tagged with POS.

    Returns:
    -------
    tagged : list of tuples
        List of tuples where each tuple is (word, POS tag).
    """
    tagged = nltk.pos_tag(tokens)
    return tagged


def tag_clouds_creation(tagged, save_path=None):
    """
    Generates and displays a word cloud for proper nouns (NNP, NNPS) from the tagged text, and saves the plot if a path is provided.

    Parameters:
    ----------
    tagged : list of tuples
        List of tuples where each tuple is (word, POS tag).
    save_path : str, optional
        Path to save the word cloud figure. If None, the figure will not be saved.

    Returns:
    -------
    None
    """
    tokens = [word for word, tag in tagged if
              word.lower() not in stopwords and word.isalpha() and tag in ['NNP', 'NNPS']]
    frequency_distribution = FreqDist(tokens)
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white'
    ).generate_from_frequencies(frequency_distribution)

    # Display the word cloud
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")

    if save_path:
        plt.savefig(save_path, format='png', dpi=300)  # Save the word cloud
        print(f"Word cloud saved to {save_path}")

    plt.show()


def txt_tranc(text):
    """
    Tokenizes the input text.

    Parameters:
    ----------
    text : list of str
        List of paragraphs to tokenize.

    Returns:
    -------
    tokens : list of str
        List of tokenized words.
    """
    tokens = []
    for paragraph in text:
        tokens.extend(nltk.word_tokenize(paragraph))
    return tokens


if __name__ == '__main__':
    file_path = 'Little_Women.html'
    with open(file_path, 'r', encoding='utf-8') as file:
        html_content = file.read()
    soup = BeautifulSoup(html_content, 'html.parser')
    paragraphs = [p.get_text() for p in soup.find_all('p')]
    paragraphs = paragraphs[4:]

    all_book_tokens = txt_tranc(paragraphs)
    tokens = text_tokenize(paragraphs)

    # Save Log Rank vs Log Frequency of All Tokens
    sorted_tokens = counting_terms(tokens=tokens, title='Log Rank vs Log Frequency of Tokens In The Book- All Tokens',
                                   save_path='.\Output\log_rank_frequency_all_tokens.png')

    top_tokens = list_of_N_top_tokens(N=20, sorted_tokens=sorted_tokens)
    print(top_tokens)

    # Save Log Rank vs Log Frequency of Tokens Without Stopwords
    tokens_without_stopwords = remove_stopwords(paragraphs)
    counting_terms(tokens=tokens_without_stopwords,
                   title='Log Rank vs Log Frequency of All Non StopWords Tokens In The Book',
                   save_path='.\Output\log_rank_frequency_non_stopwords.png')

    # Save Log Rank vs Log Frequency of Stemmed Tokens
    stemmed_words = stemming_tokens(tokens_without_stopwords)
    counting_terms(tokens=stemmed_words, title='Log Rank vs Log Frequency of All Stemmed Tokens In The Book',
                   save_path='.\Output\log_rank_frequency_stemmed_tokens.png')

    sentence = ("he never loses patience, never doubts or complains, but always hopes, and works and waits so "
                "cheerfully that one is ashamed to do otherwise before him.")
    pos_tag_print(pos_tag(nltk.word_tokenize(sentence)))

    tagged = pos_tag(all_book_tokens)

    # Save Word Cloud
    tag_clouds_creation(tagged, save_path='.\Output\word_cloud.png')

    # Pattern matching for repeated words with punctuation
    pattern = r'\b([a-zA-Z]{2,})([,.!?;:-]*)[\s]+\1\b'
    matches = set()
    for paragraph in paragraphs:
        for match in re.finditer(pattern, paragraph, re.IGNORECASE):
            word = match.group(1).lower()
            punctuation = match.group(2)
            matches.add((word, punctuation))
    # Print each unique match with punctuation
    for match in matches:
        print(f"Word: '{match[0]}', Punctuation: '{match[1]}'")

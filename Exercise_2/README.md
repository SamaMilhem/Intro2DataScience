# Text Analysis and Processing of *Little Women*

This project applies a variety of text processing techniques to analyze the book *Little Women* using Python. The main functionalities include tokenization, stopword removal, stemming, word frequency distribution (Zipf’s Law), part-of-speech tagging, and word cloud generation. It also includes a pattern-matching method for identifying repeated words followed by punctuation.

## Key Features

### 1. **Text Tokenization**

The `text_tokenize(paragraphs)` function processes the input text by tokenizing it into words, removing punctuation, and splitting hyphenated words.

- **Function:** `text_tokenize(paragraphs)`
    - **Input:** `paragraphs` – A list of paragraphs (strings) to tokenize.
    - **Output:** A list of tokenized words.

### 2. **Stopword Removal**

The `remove_stopwords(paragraphs)` function tokenizes the text and removes common stopwords, using an external list of stopwords combined with NLTK's stopwords.

- **Function:** `remove_stopwords(paragraphs)`
    - **Input:** `paragraphs` – A list of paragraphs (strings) to process.
    - **Output:** A list of tokenized words with stopwords removed.

### 3. **Word Frequency Distribution (Zipf's Law)**

The `counting_terms(tokens, title, save_path)` function plots the log rank vs. log frequency distribution of terms, demonstrating Zipf's Law.

- **Function:** `counting_terms(tokens, title, save_path)`
    - **Input:** 
        - `tokens` – A list of tokenized words.
        - `title` – Title for the plot.
        - `save_path` – Optional, a path to save the plot image.
    - **Output:** A plot showing Zipf's Law and a sorted list of token frequencies.

### 4. **Stemming Tokens**

The `stemming_tokens(tokens)` function applies the Porter Stemmer to reduce words to their root forms.

- **Function:** `stemming_tokens(tokens)`
    - **Input:** `tokens` – A list of tokenized words.
    - **Output:** A list of stemmed words.

### 5. **Part-of-Speech (POS) Tagging**

The `pos_tag(tokens)` function performs part-of-speech tagging on the tokenized words.

- **Function:** `pos_tag(tokens)`
    - **Input:** `tokens` – A list of tokenized words.
    - **Output:** A list of tuples, each containing a word and its POS tag.

### 6. **Word Cloud Generation**

The `tag_clouds_creation(tagged, save_path)` function generates a word cloud for proper nouns (NNP, NNPS) from the tagged text.

- **Function:** `tag_clouds_creation(tagged, save_path)`
    - **Input:**
        - `tagged` – A list of POS-tagged tokens.
        - `save_path` – Optional, a path to save the word cloud image.
    - **Output:** Displays a word cloud and saves it if a path is provided.

## Usage

1. Load the text (HTML format) and tokenize it using `text_tokenize()`.
2. Perform stopword removal using `remove_stopwords()`.
3. Plot and analyze word frequency distribution using `counting_terms()`.
4. Stem the tokens using `stemming_tokens()`.
5. Tag the tokens for part-of-speech using `pos_tag()`.
6. Generate a word cloud of proper nouns using `tag_clouds_creation()`.

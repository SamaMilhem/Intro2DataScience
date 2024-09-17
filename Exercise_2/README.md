# 📖✨ **Text Analysis and Processing of *Little Women*** 📚

This project applies various **text processing techniques** to analyze the book *Little Women* using Python. The main features include **tokenization**, **stopword removal**, **stemming**, **word frequency distribution (Zipf’s Law)**, **part-of-speech tagging**, and **word cloud generation**. Additionally, a pattern-matching method is included for identifying repeated words followed by punctuation.

---

## 🌟 **Key Features**

### 1️⃣ **Text Tokenization** 🔠

The `text_tokenize(paragraphs)` function breaks down the input text into smaller units (tokens), removes punctuation, and splits hyphenated words.

- **Function:** `text_tokenize(paragraphs)`
    - **Input:** `paragraphs` – A list of paragraphs (strings) to tokenize.
    - **Output:** A list of tokenized words.

### 2️⃣ **Stopword Removal** 🛑

The `remove_stopwords(paragraphs)` function removes common, non-informative words (stopwords), using both a custom list and **NLTK's** stopwords.

- **Function:** `remove_stopwords(paragraphs)`
    - **Input:** `paragraphs` – A list of paragraphs (strings) to process.
    - **Output:** A list of tokenized words with stopwords removed.

### 3️⃣ **Word Frequency Distribution (Zipf’s Law)** 📉

The `counting_terms(tokens, title, save_path)` function visualizes the log rank vs. log frequency distribution of terms, demonstrating **Zipf's Law**.

- **Function:** `counting_terms(tokens, title, save_path)`
    - **Input:** 
        - `tokens` – A list of tokenized words.
        - `title` – Title for the plot.
        - `save_path` – Optional, a path to save the plot image.
    - **Output:** A plot showing Zipf's Law and a sorted list of token frequencies.

### 4️⃣ **Stemming Tokens** 🌿

The `stemming_tokens(tokens)` function reduces words to their root forms using the **Porter Stemmer**.

- **Function:** `stemming_tokens(tokens)`
    - **Input:** `tokens` – A list of tokenized words.
    - **Output:** A list of stemmed words.

### 5️⃣ **Part-of-Speech (POS) Tagging** 🏷️

The `pos_tag(tokens)` function assigns part-of-speech tags to the tokenized words, identifying nouns, verbs, adjectives, and more.

- **Function:** `pos_tag(tokens)`
    - **Input:** `tokens` – A list of tokenized words.
    - **Output:** A list of tuples, each containing a word and its corresponding POS tag.

### 6️⃣ **Word Cloud Generation** ☁️

The `tag_clouds_creation(tagged, save_path)` function creates a **word cloud** from proper nouns (e.g., names of people or places) based on part-of-speech tagging.

- **Function:** `tag_clouds_creation(tagged, save_path)`
    - **Input:**
        - `tagged` – A list of POS-tagged tokens.
        - `save_path` – Optional, a path to save the word cloud image.
    - **Output:** Displays a word cloud and saves it if a path is provided.

---

## 🚀 **Usage**

1. **Load the text** (HTML format) and tokenize it using `text_tokenize()`:
    ```python
    tokens = text_tokenize(paragraphs)
    ```
2. **Remove stopwords**:
    ```python
    tokens_clean = remove_stopwords(tokens)
    ```
3. **Plot and analyze** word frequency distribution using **Zipf’s Law**:
    ```python
    counting_terms(tokens_clean, "Word Frequency in Little Women", save_path="word_freq.png")
    ```
4. **Stem the tokens**:
    ```python
    stemmed_tokens = stemming_tokens(tokens_clean)
    ```
5. **Tag for part-of-speech**:
    ```python
    tagged_tokens = pos_tag(stemmed_tokens)
    ```
6. **Generate a word cloud** of proper nouns:
    ```python
    tag_clouds_creation(tagged_tokens, save_path="wordcloud.png")
    ```


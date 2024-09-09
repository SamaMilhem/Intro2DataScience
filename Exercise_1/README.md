# TF-IDF, PageRank, and K-Means for Fruit Wikipedia Pages

This project implements several algorithms, including **TF-IDF**, **PageRank**, and **K-Means**, to analyze Wikipedia pages for various fruits. The goal is to extract important information and classify fruits based on specific features.

## Project Structure

### 1. **TF-IDF Directory**
This directory contains the implementation of the **TF-IDF (Term Frequency-Inverse Document Frequency)** algorithm. TF-IDF measures the importance of words across documents, allowing you to identify the most relevant terms in each Wikipedia page.

- **Function: `tfidf(df, terms)`**
    - **Input:**
      - `df`: A DataFrame where each row corresponds to a document, and a specified column contains the text of the documents.
      - `terms`: A list of terms for which the TF-IDF scores will be calculated.
    - **Output:** Returns a DataFrame with the TF-IDF scores for the given terms across the documents.
  
### 2. **PageRank Directory**

#### a. **Web Crawler**

The `fruitcrawl(fruit, link)` function is used to crawl Wikipedia pages related to specific fruits. It extracts the main content text from each page and stores the result in a JSON file.

- **Function: `fruitcrawl(fruit, link)`**
    - **Input:**
      - `fruit`: A string representing the name of the fruit (e.g., "Apple").
      - `link`: A string representing the URL of the Wikipedia page for that fruit.
    - **Output:** The function extracts the main content text of the Wikipedia page and saves it in a JSON file.

#### b. **PageRank Text Summarization**

The `textsum()` function implements a **PageRank-based text summarization** technique that ranks sentences based on their importance.

- **Function: `textsum()`**
    - **Input:** None
    - **Output:** Summarizes the text by ranking sentences according to their relevance using the PageRank algorithm.

### 3. **Combining TF-IDF and PageRank**

For each fruit:
1. The **TF-IDF** algorithm is used to compute the importance of each word in the text.
2. The top 3 words with the highest TF-IDF scores are identified.
3. A new DataFrame is created with columns corresponding to each of these top 3 words, where each row represents the TF-IDF score of the word for the respective fruit.
4. The **PageRank** summarization is used to extract key information from the text.

- **Function: `top_words()`**
    - This function runs the TF-IDF algorithm, selects the top 3 words, and adds them as columns to the DataFrame. The TF-IDF scores for these words are saved for each fruit. The results are also exported to a CSV file.

### 4. **K-Means Directory**

The **K-Means** directory contains an implementation of the **K-Means clustering algorithm**, which groups the fruits based on specific features into `k` clusters.

- **Function: `runKmeansAlgorithm(dataset, k, calculate_distance_function, rearrange_clusters_function, initiate_clusters_func)`**
    - **Input:**
      - `dataset`: A dataset containing the features of different fruits.
      - `k`: The number of clusters to form.
      - `calculate_distance_function`: A function to calculate the distance between data points (e.g., Euclidean distance).
      - `rearrange_clusters_function`: A function to rearrange data points into clusters based on the calculated distances.
      - `initiate_clusters_func`: A function to initialize the clusters.
    - **Output:** Groups the dataset into `k` clusters using the specified distance and rearrangement functions.

### Helper Functions for K-Means:

- **`calculate_distance_function`:** Computes the distance between data points.
- **`rearrange_clusters_function`:** Rearranges data points into clusters based on distance.
- **`initiate_clusters_func`:** Initializes the clusters before the K-Means algorithm starts.

You can use these helper functions directly or modify them to suit your needs.

### Usage

1. To crawl the Wikipedia page for a fruit and extract the text:
    ```python
    fruitcrawl("Apple", "https://en.wikipedia.org/wiki/Apple")
    ```

2. To run the TF-IDF algorithm on a DataFrame containing the fruit texts:
    ```python
    df = ... # Load your DataFrame
    terms = ["apple", "fruit", "tree"] # List of terms
    tfidf(df, terms)
    ```

3. To summarize the text using the PageRank algorithm:
    ```python
    textsum()
    ```

4. To extract the top 3 words with the highest TF-IDF scores for each fruit and save the results:
    ```python
    top_words()
    ```

5. To run the K-Means algorithm on a fruits dataset with specific features:
    ```python
    runKmeansAlgorithm(dataset, k, calculate_distance_function, rearrange_clusters_function, initiate_clusters_func)
    ```

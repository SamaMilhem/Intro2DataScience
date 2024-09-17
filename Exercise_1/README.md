# 🍎 **TF-IDF, PageRank, and K-Means for Fruit Wikipedia Pages** 🍌

This project applies key data science algorithms—**TF-IDF**, **PageRank**, and **K-Means**—to analyze Wikipedia pages for various fruits. The goal is to extract meaningful insights and classify fruits based on unique textual features.

---

## 🗂️ **Project Structure**

### 1️⃣ **TF-IDF Directory** 📝

The **TF-IDF (Term Frequency-Inverse Document Frequency)** algorithm measures the importance of words across multiple documents, allowing us to identify the most relevant terms in each Wikipedia page.

- **Function: `tfidf(df, terms)`**
    - **Input:**
      - `df`: A DataFrame where each row represents a document, and a specific column contains the text.
      - `terms`: A list of terms for which TF-IDF scores will be calculated.
    - **Output:** A DataFrame containing the TF-IDF scores for the given terms across all documents.

---

### 2️⃣ **PageRank Directory** 🌐

#### a. **Web Crawler** 🕸️

The `fruitcrawl(fruit, link)` function crawls Wikipedia pages for specific fruits, extracting the main content and storing it in a JSON file.

- **Function: `fruitcrawl(fruit, link)`**
    - **Input:**
      - `fruit`: The name of the fruit (e.g., "Apple").
      - `link`: The URL of the Wikipedia page for the fruit.
    - **Output:** Extracts the main content from the page and saves it as a JSON file.

#### b. **PageRank-Based Text Summarization** 📊

The `textsum()` function ranks sentences based on their importance using the **PageRank algorithm**, providing a concise summary of the text.

- **Function: `textsum()`**
    - **Input:** None.
    - **Output:** Summarizes the text by ranking sentences using the PageRank algorithm.

---

### 3️⃣ **Combining TF-IDF and PageRank** 🔗

This combined approach provides a powerful analysis of each fruit:

1. **TF-IDF** is used to compute the importance of each word in the text.
2. The top 3 words with the highest TF-IDF scores are identified.
3. A new DataFrame is created with columns representing the top 3 words, and their respective TF-IDF scores are stored for each fruit.
4. **PageRank** summarization extracts key information from the text.

- **Function: `top_words()`**
    - Runs the TF-IDF algorithm, identifies the top 3 words, and saves the results to a CSV file.

---

### 4️⃣ **K-Means Directory** 📈

The **K-Means clustering algorithm** is implemented to group fruits based on shared features into `k` clusters.

- **Function: `runKmeansAlgorithm(dataset, k, calculate_distance_function, rearrange_clusters_function, initiate_clusters_func)`**
    - **Input:**
      - `dataset`: A dataset containing features of different fruits.
      - `k`: The number of clusters to form.
      - `calculate_distance_function`: A function to calculate the distance between data points.
      - `rearrange_clusters_function`: A function to rearrange data points into clusters.
      - `initiate_clusters_func`: A function to initialize clusters.
    - **Output:** Groups the dataset into `k` clusters based on the specified distance and rearrangement functions.

#### ⚙️ **Helper Functions for K-Means:**

- **`calculate_distance_function`:** Computes the distance between data points.
- **`rearrange_clusters_function`:** Rearranges data points into clusters based on the calculated distances.
- **`initiate_clusters_func`:** Initializes the clusters before the K-Means algorithm starts.

---

## 🚀 **Usage Instructions**

### 1. Web Crawling 🕵️‍♂️

To crawl a Wikipedia page for a fruit and extract the text:

```python
fruitcrawl("Apple", "https://en.wikipedia.org/wiki/Apple")
```

### 2. TF-IDF Analysis 📊

To run the TF-IDF algorithm on a DataFrame containing the fruit texts:

```python
df = ...  # Load your DataFrame
terms = ["apple", "fruit", "tree"]  # Specify terms
tfidf(df, terms)
```

### 3. PageRank Summarization 🔝

To summarize the text using the PageRank algorithm:

```python
textsum()
```

### 4. Extract Top TF-IDF Words 📄

To extract the top 3 words with the highest TF-IDF scores for each fruit and save the results:

```python
top_words()
```

### 5. K-Means Clustering 🤖

To run the K-Means algorithm on a fruits dataset:

```python
runKmeansAlgorithm(dataset, k, calculate_distance_function, rearrange_clusters_function, initiate_clusters_func)
```

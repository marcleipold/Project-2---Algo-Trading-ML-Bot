# Project-2---Algo-Trading-ML-Bot

## Vader Sentiment Analysis
For my part of the algo trading bot, I decided to create a Sentiment analysis of Tweets referencing Bitcoin during the period of February 2021 to August 2021. For this project, I decided to use the Vader library to perform sentiment analysis on a set of tweets about the hashtag #BTC.

### Prerequisites
Before you begin, make sure that you have the following packages installed:

- pandas
- nltk
- hvplot
- 
To install the required packages, run the following command in your terminal:

```
pip install pandas nltk hvplot
```

### Running the Analysis
1. Clone this repository to your local machine
2. Open the vader_sentiment_analysis.ipynb Jupyter Notebook
3. Run each cell in the notebook to perform the sentiment analysis
4. The final result of the analysis will be saved to a CSV file and visualized in a histogram plot


### Data
The data used for this analysis is a set of tweets about the hashtag #BTC. The tweets were collected using the Twitter API and pre-processed to remove irrelevant information.

### Sentiment Analysis with Vader
The Vader library is used to perform sentiment analysis on the text data. The library uses a lexicon of words and emoticons to determine the sentiment of a given text. The resulting sentiment scores are reported as negative, neutral, positive, and compound scores, where the compound score is a normalized score between -1 and 1 that indicates the overall sentiment of the text.

### Visualizing the Results
The results of the sentiment analysis are visualized using the hvplot library. The histogram plot shows the distribution of the Vader compound scores for each date. This provides an overall picture of the sentiment towards the hashtag #BTC on different dates.

### Conclusion
Vader Sentiment Analysis is a quick and effective way to determine the sentiment of a given text. This project demonstrates how to perform sentiment analysis on a set of tweets about the hashtag #BTC and visualize the results.

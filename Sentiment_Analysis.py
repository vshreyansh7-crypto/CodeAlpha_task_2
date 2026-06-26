import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv("reviews.csv")

# Check if the dataset contains the Review column
if "Review" not in df.columns:
    raise Exception("CSV file must contain a column named 'Review'")

# -----------------------------
# Sentiment Analysis Function
# -----------------------------
def get_sentiment(text):
    text = str(text)

    polarity = TextBlob(text).sentiment.polarity

    if polarity > 0.1:
        return "Positive"
    elif polarity < -0.1:
        return "Negative"
    else:
        return "Neutral"

# -----------------------------
# Polarity Score Function
# -----------------------------
def get_polarity(text):
    return TextBlob(str(text)).sentiment.polarity

# -----------------------------
# Apply Sentiment Analysis
# -----------------------------
df["Polarity"] = df["Review"].apply(get_polarity)
df["Sentiment"] = df["Review"].apply(get_sentiment)

# -----------------------------
# Save Results
# -----------------------------
df.to_csv("results.csv", index=False)

print("\nSentiment Analysis Completed Successfully!\n")
print(df)

# -----------------------------
# Sentiment Counts
# -----------------------------
counts = df["Sentiment"].value_counts()

print("\nSummary")
print(counts)

# -----------------------------
# Bar Chart
# -----------------------------
plt.figure(figsize=(7,5))

counts.plot(kind="bar")

plt.title("Sentiment Analysis")
plt.xlabel("Sentiment")
plt.ylabel("Number of Reviews")

plt.tight_layout()
plt.savefig("sentiment_bar_chart.png", dpi=300)
plt.show()

# -----------------------------
# Pie Chart
# -----------------------------
plt.figure(figsize=(6,6))

counts.plot(
    kind="pie",
    autopct="%1.1f%%",
    startangle=90
)

plt.ylabel("")
plt.title("Sentiment Distribution")

plt.tight_layout()
plt.savefig("sentiment_pie_chart.png", dpi=300)
plt.show()

print("\nFiles Generated Successfully:")
print("1. results.csv")
print("2. sentiment_bar_chart.png")
print("3. sentiment_pie_chart.png")

review = input("Enter your review: ")
def analyze_review(review):
    polarity = TextBlob(review).sentiment.polarity

    if polarity > 0:
        sentiment = "Positive"
    elif polarity < 0:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"

    print("\nReview:", review)
    print("Sentiment:", sentiment)
    print("Polarity Score:", round(polarity,2))
analyze_review(review)
df["Sentiment"] = df["Review"].apply(get_sentiment)

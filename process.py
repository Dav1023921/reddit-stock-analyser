import praw
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

nltk.download("vader_lexicon")

sia = SentimentIntensityAnalyzer()

sentiment_array = []
sentiment_data = []

keywords = ["nvidia", "nvda", "nvd"]

reddit = praw.Reddit(
    client_id="W3vFbKNOON5oyudHnAu_0Q",
    client_secret="yk_JcB281iRtBjdv9l8VQha6Ly6_xA",
    user_agent="Comment Extraction (by u/One_Relation8674)"

)
def average_sentiment():
    average_sentiment = 0
    for value in sentiment_array:
        average_sentiment += value["compound"]
    return average_sentiment

def get_sentiment_data():
    return sentiment_data


subreddit = reddit.subreddit("wallstreetbets")

comments = subreddit.comments(limit=1000)

for comment in comments:
    if any(keyword in comment.body.lower() for keyword in keywords):
        sentiment = sia.polarity_scores(comment.body)
        print(f"Sentiment Scores: {sentiment}")
        print(f"Comment: {comment.body}\n")
        sentiment_array.append(sentiment)
    else:
        print("none")


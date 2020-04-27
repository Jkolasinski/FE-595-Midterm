#Jakub Kolasinski
#FE 595 Midterm
#NLP functions

import nltk
from collections import Counter
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from nltk.sentiment.vader import SentimentIntensityAnalyzer


def word_cloud(tweets):
    stopwords = set(STOPWORDS)
    all_words = ' '.join([text for text in tweets])
    wordcloud = WordCloud(
        background_color='white',
        stopwords=stopwords,
        width=1600,
        height=800,
        random_state=21,
        colormap='jet',
        max_words=50,
        max_font_size=200).generate(all_words)
    plt.figure(figsize = (8, 8), facecolor = None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad = 0)
    plt.show()



def sentiment(tweets):
    sid = SentimentIntensityAnalyzer()
    for tweet in tweets:
        scores = sid.polarity_scores(tweet)
    
        for key in sorted(scores):
            print('{0}: {1} '.format(key, scores[key]), end='')
    
        if scores["compound"] >= 0.05:
            print("\npositive\n")
    
        elif scores["compound"] <= -0.05:
            print("\nnegative\n")
        else:
            print("\nneutral\n")
            
            
            
def poscount(tweets):
    for tweet in tweets:
        lower_case = tweet.lower()
        tokens = nltk.word_tokenize(lower_case)
        tags = nltk.pos_tag(tokens)
        counts = Counter( tag for word,  tag in tags)
        print(counts)
        

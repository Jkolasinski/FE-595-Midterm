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
    return plt



def sentiment(tweets):
    sid = SentimentIntensityAnalyzer()
    results = ''
    sent_list = []
    for tweet in tweets:
        scores = sid.polarity_scores(tweet)
    
        for key in sorted(scores):
            sent_list.append('{0}: {1} '.format(key, scores[key]))
    
        if scores["compound"] >= 0.05:
            sent_list.append('Positive')
    
        elif scores["compound"] <= -0.05:
            sent_list.append('Negative')
        else:
            sent_list.append('Neutral')
    results = '<br/>'.join(sent_list)
    return results
            
            
            
def poscount(tweets):
    results = ''
    count_list = []
    for tweet in tweets:
        lower_case = tweet.lower()
        count_list.append(lower_case)
        tokens = nltk.word_tokenize(lower_case)
        tags = nltk.pos_tag(tokens)
        counts = dict(Counter( tag for word,  tag in tags))
        for key in list(counts.keys()):
            count_list.append(str(key)+': '+str(counts[key]))
    results = '<br/>'.join(count_list)
    return results
        

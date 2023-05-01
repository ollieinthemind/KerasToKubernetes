import nltk

nltk.download('wordnet')

# this will be the text document we will analyze
mytext = "We are studying Machine Learning. Our Model learns patterns in data. This learning helps it to predict on new data."
print("ORIGINAL TEXT = " , mytext)
print('----------------------')

# convert text to lowercase
mytext = mytext.lower()

# first we will tokenize the text into word tokens
word_tokens = nltk.word_tokenize(mytext)
print("WORD TOKENS = ", word_tokens)
print('---------------------')

# and then extract sentences
sentence_tokens = nltk.sent_tokenize(mytext)
print("Sentence Tokens = ", sentence_tokens)
print('----------------')

#removing common stop words
stp_words = ["is", "a", "our", "on", ".", "we", "are", "this", "of", "and", "from", "to", "it", "in"]
print("STOP WORDS = ", stp_words)
print('-------------------')

#define clean up tokens array
clean_tokens = []

#remove stop words from our word_tokens
for token in word_tokens:
    if token not in stp_words:
        clean_tokens.append(token)

print("CLEANED WORD TOKENS = ", clean_tokens)
print('-----------------')

from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
from nltk.stem import PorterStemmer
stemmer = PorterStemmer()

# define cleaned up and lemmatized tokens array

clean_lemma_tokens = []
clean_stem_tokens = []

#remove stop words from our word_tokens
for token in clean_tokens:
    clean_stem_tokens.append(stemmer.stem(token))
    clean_lemma_tokens.append(lemmatizer.lemmatize((token)))

print("CLEANED STEMMED TOKENS = ", clean_stem_tokens)
print('-------------------')

print("CLEANED LEMMATIZED TOKENS = ", clean_lemma_tokens)
print('-------------------')

#get frequency distribution of words

freq_lemma = nltk.FreqDist(clean_lemma_tokens)
freq_stem = nltk.FreqDist(clean_stem_tokens)

#import plotting library
import matplotlib.pyplot as plt

chart_fontsize = 10

#plot the frequency chart
plt.figure(figsize=(10,5))
plt.tick_params(labelsize=chart_fontsize)
plt.title('Cleaned and Stemmed Words', fontsize=chart_fontsize)
plt.xlabel('Word Tokens', fontsize=chart_fontsize)
plt.ylabel('Frequency (Counts)', fontsize=chart_fontsize)
freq_stem.plot(20,cumulative=False)
plt.show()

# plot the frequency chart
plt.figure(figsize=(20,10))
plt.tick_params(labelsize=chart_fontsize)
plt.title('Cleaned and Lemmatized Words', fontsize=chart_fontsize)
plt.xlabel('Word Tokens', fontsize=chart_fontsize)
plt.ylabel('Frequency (Counts)', fontsize=chart_fontsize)
freq_lemma.plot(20, cumulative=False)
plt.show()
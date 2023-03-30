import nltk


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


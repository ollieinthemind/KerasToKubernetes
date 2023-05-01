import nltk


# define the sentence that will be analyzed
mytext = "AI is the new electricity. AI is poised to start a large transformation on many industries."
# we will first tokenize the text
word_tk = nltk.word_tokenize(mytext)
words = [w.lower() for w in word_tk]


# create a vocabulary of all relevant words
vocab = sorted(set(words))

print("VOCABULARY = ", vocab)
print('----------------------')


# create one hot encoded vectors for each word
for myword in vocab:
    test_1hot = [0]*len(vocab)
    test_1hot[vocab.index(myword)] = 1
    print("ONE HOT VECTOR FOR '%s' = "%myword, test_1hot)
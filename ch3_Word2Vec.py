import nltk
from gensim.models import Word2Vec
# import the PCA library from scikit-learn
from sklearn.decomposition import PCA

#import plotting library
import matplotlib.pyplot as plt

mytext = "AI is the new electricity. AI is poised to start a large transformation on many industries."
print("ORIGINAL TEXT = ", mytext)
print("------------------------")


mytext = mytext.lower()

sentence_tokens=nltk.sent_tokenize(mytext)

print("SENTENCE TOKENS = ", sentence_tokens)
print('----------------------')

stp_words = ["is","a","our","on",".","!","we","are","this","of","and", "from","to","it","in"]

# define training data

sentences = []
for sentence in sentence_tokens:
    word_tokens = nltk.word_tokenize(sentence)

    #define cleaned up tokens array
    clean_tokens = []

    #remove stop words from our workd_tokens
    for token in word_tokens:
        if token not in stp_words:
            clean_tokens.append(token)
    sentences.append(clean_tokens)

print("TRAINING DATA = ", sentences)
print('----------')

# train a new word2vec model on our data - we will use embedding size 20

word2vec_model = Word2Vec(sentences, vector_size=20, min_count=1)

# list the vocabulary learned from our corpus
words = list(word2vec_model.wv.index_to_key)
print("VOCABULARY OF MODEL = ", words)
print('----------------------------------------')

# show the embeddings vector for some words
print("EMBEDDINGS VECTOR FOR THE WORD 'ai' = ", word2vec_model.wv.__getitem__('ai'))
print("EMBEDDINGS VECTOR FOR THE WORD 'electricity' = ", word2vec_model.wv.__getitem__('electricity'))

# build training data using word2vec model
training_data = word2vec_model.wv.vectors
print(training_data)
# use PCA to convert word vectors to 2 dimensional vectors
pca = PCA(n_components=2)
result = pca.fit_transform(training_data)

# create a scatter plot of the 2 dimensional vectors
plt.figure(figsize=(20,15))
plt.rcParams.update({'font.size': 25})
plt.title('Plot of Word embeddings from Text')
plt.scatter(result[:, 0], result[:, 1], marker="X")

# mark the words on the plot
words = list(word2vec_model.wv.index_to_key)
for i, word in enumerate(words):
    plt.annotate(word, xy=(result[i, 0], result[i, 1]))

plt.show()
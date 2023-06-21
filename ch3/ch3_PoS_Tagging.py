import nltk

nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')

# define the sentence that will be analyzed

mysentence = "Mark is working at GE"

print("SENTENCE TO ANALYZE = ", mysentence)
print('------------------')

#now we will map parts of speech (pos) for the sentence
word_tk = nltk.word_tokenize(mysentence)
pos_tags = nltk.pos_tag(word_tk)
print("PARTS OF SPEECH FOR SENTENCE = ", pos_tags)
print('--------------------')

entities = nltk.chunk.ne_chunk(pos_tags)
print("NAMED ENTITIES FOR SENTENCE = ", entities)
print('--------------------')


# define the sentence that will be analyzed
mytext = "AI is the new electricity. AI is poised to start a large transformation on many industries."


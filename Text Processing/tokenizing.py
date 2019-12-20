from nltk.tokenize import sent_tokenize,word_tokenize

# tokenizing - word tokenizer.... sentence tokenizer
# lexicon and corporas
# corporas - body of text. ex: medical journals.
# lexicon - words and their means

example_text = 'Hello Mr. Smith, how are you doing today? The weather is very good.'
# sentence tokenizer
print('Sentences tokenizer: ',sent_tokenize(example_text))
# word tokenizer
print('Words tokenizer: ', word_tokenize(example_text))


for i in word_tokenize(example_text):
    print(i)
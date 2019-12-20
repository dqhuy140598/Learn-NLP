from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize


ps = PorterStemmer()

example_words = ['python','pythoner','pythoning','pythoned','pythonly']

# for word in example_words:
#     print(ps.stem(word))


new_text = 'It is very important to be pythonly while you are pythoning with python. All pythoners have pythoned poorly at least once.'

words = word_tokenize(new_text)

for word in words:
    print(ps.stem(word))





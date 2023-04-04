# # coding: utf8
# from spacy.lang.ne.stop_words import STOP_WORDS
# from spacy.lang.ne import Nepali 
# from spacy.lang.ne.examples import sentences

# nlp = Nepali()

# def tokenize(texts):
#     docs = nlp.pipe(texts)
#     for doc in docs:
#         tests = [(w.text) for w in doc] #(w.text,w.pos)
#     return tests

# # for sent in sentences:
# #     tests = tokenize([sent])
# #     print(tests)

# # with open("testing.txt", 'w', encoding="utf-8") as f:
# #     for word in tests:
# #         f.write(str(word))
# #     f.close()

import string

class Tokenizer:
    def __init__(self):
        pass

    def sentence_tokenize(self, text):
        """This function tokenize the sentences
        
        Arguments:
            text {string} -- Sentences you want to tokenize
        
        Returns:
            sentence {list} -- tokenized sentence in list
        """
        sentences = text.split(u"।")
        sentences = [sentence.translate(str.maketrans('', '', string.punctuation)) for sentence in sentences]
        return sentences

    def word_tokenize(self, sentence, new_punctuation=[]):
        """This function tokenize with respect to word
        
        Arguments:
            sentence {string} -- sentence you want to tokenize
            new_punctuation {list} -- more punctutaion for tokenizing  default ['।',',',';','?','!','—','-']
        
        Returns:
            list -- tokenized words
        """
        punctuations = ['।', ',', ';', '?', '!', '—', '-', '.']
        if new_punctuation:
            punctuations = set(punctuations + new_punctuation)

        for punct in punctuations:
            sentence = ' '.join(sentence.split(punct))

        return sentence.split()

    def character_tokenize(self, word):
        """ Returns the tokenization in character level.
        
        Arguments:
            word {string} -- word to be tokenized in character level.
        
        Returns:
            [list] -- list of characters.
        """
        try:
            import icu

        except:
            print("please install PyICU")
        
        temp_ = icu.BreakIterator.createCharacterInstance(icu.Locale())
        temp_.setText(word)
        char = []
        i = 0
        for j in temp_:
            s = word[i:j]
            char.append(s)
            i = j

        return char

    def __str__(self):
        return "Helps to tokenize content written in Nepali language."
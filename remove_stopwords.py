import ast

class Stopwords:
    """This class helps in removing Nepali stopwords."""
    def __init__(self):
        pass

    def remove_stopwords(self,token):
        """This function remove stopwords from text
        
        Arguments:
        sentence {string} -- sentence you want to remove stopwords
        Returns:
            list -- token words
        """
        f = open("stopwords.txt",'r', encoding='utf-8')
        stopwords = f.read()
        stopwords = ast.literal_eval(stopwords)

        word_without_stopword=[]
        for word in token:
            if word not in stopwords:
                word_without_stopword.append(word)
            
        return word_without_stopword
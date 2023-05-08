import pandas as pd
import ast
from tokenizer import Tokenizer
from remove_stopwords import Stopwords
from stemmer import Stem
from inputValidator import InputValidator
import time

def create_dataset():
    file_content = pd.read_csv("../Summarizer/dataset/updatedCSV.csv")
    file_content = file_content.dropna()
    titles = []
    contents = []
    for i in file_content['title'][:100000]:#[399096:]:
        i = str(i).replace(u"\xa0",' ')
        i = i.replace(u"\xc2\xa0",' ')
        i = i.strip()
        # print(i)
        # exit()
        titles.append(i)
    # print(titles)
    # exit()
    # print('------------------------')
    for i in file_content['content'][:100000]:#[399096:]:
        i = str(i).replace(u"\xa0",' ')
        i = i.replace(u"\xc2\xa0",' ')
        i = i.strip()
        contents.append(i)
    # print(contents)
    from sklearn.model_selection import train_test_split
    x_tr,x_val,y_tr,y_val=train_test_split(contents,titles,test_size=0.2,random_state=0,shuffle=True)
    # print(len(x_tr)) #323276
    # print(len(x_val)) #80820
    # print(len(y_tr)) #323276
    # print(len(y_val)) #80820
    create_files('sumdata/train/train.article.txt', x_tr, 'content')
    create_files('sumdata/train/valid.article.filter.txt', x_val, 'content')
    create_files('sumdata/train/train.title.txt', y_tr, 'title')
    create_files('sumdata/train/valid.title.filter.txt', y_val, 'title')


def create_files(filename, data, val_type):
    with open(filename, 'w', encoding='utf-8') as f:
        tokenizer = Tokenizer()
        start_time = time.time()
        
        for index, i in enumerate(data):
            if index % 1000 == 0:
                elapsed_time = time.time() - start_time
                print(f"Processing item {index}/{len(data)}. Elapsed time: {elapsed_time:.2f} seconds.")
           
            # f.write(i + '\n')
        for i in data:
            # print(i)
            # exit()
            if val_type=='content':
                # val = ast.literal_eval(i)
                val=i.split()
                processed_content = []
                for j in val:
                    validateobj = InputValidator(j)
                    tokenized_title = tokenizer.word_tokenize(validateobj.validate_to_var())
                    removed_stopwords_title = Stopwords().remove_stopwords(tokenized_title)
                    stemmed_title = Stem().rootify(removed_stopwords_title)
                    stemmed_title = " ".join(stemmed_title)
                    processed_content.append(stemmed_title)
                    # f.write(str(stemmed_title).rstrip('\n'))
                f.write("".join(processed_content)+"\n")
            elif val_type=='title':
                validateobj = InputValidator(i)
                tokenized_title = tokenizer.word_tokenize(validateobj.validate_to_var())
                # print(tokenized_title)
                removed_stopwords_title = Stopwords().remove_stopwords(tokenized_title)
                # print(removed_stopwords_title)
                stemmed_title = Stem().rootify(removed_stopwords_title)
                stemmed_title = " ".join(stemmed_title)
                # print(stemmed_title)
                # exit()
                f.write(str(stemmed_title+'\n'))
            else:
                print('Incorrect parameter : should be either content or title')
    
create_dataset()
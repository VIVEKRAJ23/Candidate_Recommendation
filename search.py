# from distutils.errors import LibError
import glob
import os
import warnings
import textract
# import requests
from flask import (Flask, json, Blueprint, jsonify, redirect, render_template, request,
                   url_for)
# from gensim.summarization import summarize
# from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
# from sklearn.neighbors import NearestNeighbors
# from werkzeug import secure_filename

import pdf2txt as pdf
# import PyPDF2


warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

app = Flask(__name__)


app.config['UPLOAD_FOLDER'] = 'Original_Resumes/'
app.config['ALLOWED_EXTENSIONS'] = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])


class ResultElement:
    def __init__(self, rank, filename, count):
        self.rank = rank
        self.filename = filename
        self.count = count


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']




import re, string, unicodedata
import nltk
# import contractions
import inflect
# from bs4 import BeautifulSoup
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, WordNetLemmatizer

def remove_non_ascii(words):
    """Remove non-ASCII characters from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        new_words.append(new_word)
    return new_words

def to_lowercase(words):
    """Convert all characters to lowercase from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = word.lower()
        new_words.append(new_word)
    return new_words

def remove_punctuation(words):
    """Remove punctuation from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = re.sub(r'[^\w\s]', '', word)
        if new_word != '':
            new_words.append(new_word)
    return new_words

def replace_numbers(words):
    """Replace all interger occurrences in list of tokenized words with textual representation"""
    p = inflect.engine()
    new_words = []
    for word in words:
        if word.isdigit():
            new_word = p.number_to_words(word)
            new_words.append(new_word)
        else:
            new_words.append(word)
    return new_words

def remove_stopwords(words):
    """Remove stop words from list of tokenized words"""
    new_words = []
    for word in words:
        # print(word)
        if word not in stopwords.words('english'):
            new_words.append(word)
    return new_words

def stem_words(words):
    """Stem words in list of tokenized words"""
    stemmer = LancasterStemmer()
    stems = []
    for word in words:
        stem = stemmer.stem(word)
        stems.append(stem)
    return stems

def lemmatize_verbs(words):
    """Lemmatize verbs in list of tokenized words"""
    lemmatizer = WordNetLemmatizer()
    lemmas = []
    for word in words:
        lemma = lemmatizer.lemmatize(word, pos='v')
        lemmas.append(lemma)
    return lemmas

def normalize(words):
    words = remove_non_ascii(words)
    words = to_lowercase(words)
    words = remove_punctuation(words)
    # words = replace_numbers(words)
    words = remove_stopwords(words)
    # words = stem_words(words)
    # words = lemmatize_verbs(words)
    return words

def getfilepath(loc):
    # temp = str(loc)
    # temp = temp.replace('\\', '/')
    temp = str(loc).split('\\')
    return temp[-1]
    # return temp

def getfilename(loc):
    temp = str(loc).split('\\')[1].split('.')
    return temp[-3]

def extractTextFromDocx(filename):
    a = textract.process(filename)
    a = a.replace(b'\n',  b' ')
    a = a.replace(b'\r',  b' ')
    b = str(a)
    return b

def getdocxfilepath(loc):
    temp = str(loc).split('\\')
    # print("temp: ",temp)
    return temp[-1][:-5]


def docx_pfd_cleaner(text):

    text = text.replace("/"," ").replace('+','qwe')
    sentences = nltk.sent_tokenize(text)
    final_text = []

    for sen in sentences:
        tokens = word_tokenize(sen)
        words = [word for word in tokens if word.isalnum() or word.replace('.', '', 1).isdigit()]
        final_text.append(words)

    final_text = [ele for ele in final_text if ele != [] and len(ele) != 1]
    final_temp = []
    for sentence in final_text:
        temp = []
        for word in sentence:

            if word.lower() not in stopwords.words('english'):
                temp.append(word.lower())
            
        if temp != []:
            final_temp.append(temp)

    resume_text = []
    for sen in final_temp:
        resume_text.append(" ".join(sen).replace('qwe','+'))
    

    return resume_text


def res(pattern):
    Final_Array = []
    

    def KMP(pattern, text):
        a = len(text)
        b = len(pattern)
        prefix_arr = get_prefix_arr(pattern, b)
    
        initial_point = []
        m = 0
        n = 0
    
        while m != a:
        
            if text[m] == pattern[n]:
                m += 1
                n += 1
        
            else:
                n = prefix_arr[n-1]
        
            if n == b:
                initial_point.append(m-n)
                n = prefix_arr[n-1]
            elif n == 0:
                m += 1
    
        return initial_point

    def get_prefix_arr(pattern, b):
        prefix_arr = [0] * b
        n = 0
        m = 1
        while m != b:
            if pattern[m] == pattern[n]:
                n += 1
                prefix_arr[m] = n
                m += 1
            elif n != 0:
                    n = prefix_arr[n-1]
            else:
                prefix_arr[m] = 0
                m += 1
        return prefix_arr
            
    
    # def spellCorrect(string):
    #     words = string.split(" ")
    #     correctWords = []
    #     for i in words:
    #         correctWords.append(spell(i))
    #     return " ".join(correctWords)
    
    def semanticSearch(searchString, searchSentencesList):
        result = None
        # searchString = spellCorrect(searchString)
        bestScore = 0
        for i in searchSentencesList:
            score = KMP(searchString, i)
            print(score , i[0:100])
            print("")
            temp1 = [score]
            Final_Array.extend(temp1)
            if score > bestScore:
                bestScore = score
                result = i
        return result
    
    app.config['UPLOAD_FOLDER'] = 'Original_Resumes/'
    app.config['ALLOWED_EXTENSIONS'] = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

    def allowed_file(filename):
        return '.' in filename and \
            filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']





    Ordered_list_Resume = []

    Resumes_File_Names = []
    # Resume_Vector = []
    # Ordered_list_Resume = []
    # Ordered_list_Resume_Score = []
    LIST_OF_PDF_FILES = []
    LIST_OF_FILES_DOCX = []
    LIST_OF_FILES = []

   
    for file in glob.glob("./Original_Resumes/*.pdf"):
        LIST_OF_PDF_FILES.append(file)
    for file in glob.glob('./Original_Resumes/*.docx'):
        LIST_OF_FILES_DOCX.append(file)
    LIST_OF_FILES = 0
    LIST_OF_FILES = LIST_OF_FILES_DOCX + LIST_OF_PDF_FILES
    print(LIST_OF_FILES)
    if os.path.isfile("Old_files.txt") and os.stat("Old_files.txt").st_size != 0:
        new_file=[]
        with open("Old_files.txt",'r') as file:
            file.seek(0)
            old_file = file.read().splitlines()

        for files in os.walk(r'./Original_Resumes'):
            for filename in files[2]:
                new_file.append(filename) 
            
        if new_file == old_file:
            Same_content = True
            print("Same Content")
            
        else:
            Same_content = False
            print("Different Content")
            with open("Old_files.txt",'w+') as file:
                for filename in files[2]:
                    file.write("%s\n" %filename)

                
    else:
        Same_content = False
        with open("Old_files.txt",'w+') as file:
            for files in os.walk(r'./Original_Resumes'):
                for filename in files[2]:
                    file.write("%s\n" %filename)
        print("File Created")
   
    

    if Same_content == False:
        files = glob.glob('./Parsed_Resumes/*')
        for f in files:
            os.remove(f)


        print("Total Files to Parse\t :" , len(LIST_OF_FILES))
        print("DOCX files to Parse\t :" , len(LIST_OF_FILES_DOCX))
        print("PDF Files to Parse\t :" , len(LIST_OF_PDF_FILES))
        
        print("####### PARSING ########")
        
        for i in LIST_OF_PDF_FILES:
            print(i)
            pdf.extract_text([str(i)] , all_texts=None , output_type='text' , outfile='Parsed_Resumes\\'+getfilepath(i)+'.txt')

        for i in LIST_OF_FILES_DOCX:
            print(i)
            try:
                
                docx_text = extractTextFromDocx(i)
                
                # print("docxtxt------------------------->",docx_text)
                cleaned_text = docx_pfd_cleaner(docx_text)
                # print("cleaned_text------------------------->",cleaned_text)

                with open('./Parsed_Resumes/'+getdocxfilepath(i)+'.pdf.txt', 'w') as f:
                    f.write(". ".join(cleaned_text))
                    
                    
        
                


            except Exception as e: 
                print(e)


        print("Done Parsing.")


        # Job_Desc = 0
        # JD = []
        # for file in glob.glob("./Job_Description/*.txt"):
        #     JD.append(file)


        LIST_OF_TXT_FILES = []
        for file in glob.glob("./Parsed_Resumes/*.txt"):
            LIST_OF_TXT_FILES.append(file)

        Final_Array=[]
        Resumes_File_Names=[]
        for i in LIST_OF_TXT_FILES:
            Ordered_list_Resume.append(i)
            f = open(i , encoding="utf8")
            text = f.read()
            
            # text = str(text)
            # print("\n\n")
            # print(tttt)
            # print("\n\n")

            text = docx_pfd_cleaner(text)
            f.close()
            name = i.split("\\")[1].split(".")[0]
            Resumes_File_Names.append(name)
            string = str(text)
            string = string.lower()
            # pattern = "machine learning"
            pattern = pattern
            pattern = pattern.lower()
            initial_index = KMP(pattern, string)
            count=0
            for i in initial_index:
                count+=1

            Final_Array.append(count)


        
        # Z = [x for _,x in sorted(zip(Final_Array,Resumes_File_Names) , reverse=True)]
        Z = [(i,j) for i,j in sorted(zip(Final_Array,Resumes_File_Names) , reverse=True)]
        print("Z-->",Z)
        flask_return = []


        for n,i in enumerate(Z):
            name = i[1]
            count= i[0]
            rank= n+1
            res = ResultElement(rank, name, count)
            flask_return.append(res)
 
        return flask_return
    
    else:
        LIST_OF_TXT_FILES = []
        for file in glob.glob("./Parsed_Resumes/*.txt"):
            LIST_OF_TXT_FILES.append(file)

        Final_Array=[]
        Resumes_File_Names=[]
        for i in LIST_OF_TXT_FILES:
            Ordered_list_Resume.append(i)
            f = open(i , encoding="utf8")
            text = f.read()

            text = docx_pfd_cleaner(text)
            f.close()
            name = i.split("\\")[1].split(".")[0]
            Resumes_File_Names.append(name)
            string = str(text)
            string = string.lower()
            pattern = pattern
            pattern = pattern.lower()
            initial_index = KMP(pattern, string)
            count=0
            for i in initial_index:
                count+=1

            Final_Array.append(count)


        
        # Z = [x for _,x in sorted(zip(Final_Array,Resumes_File_Names) , reverse=True)]
        Z = [(i,j) for i,j in sorted(zip(Final_Array,Resumes_File_Names) , reverse=True)]
        print("Z-->",Z)
        flask_return = []

        for n,i in enumerate(Z):

            name = i[1]
            count= i[0]
            rank= n+1
            res = ResultElement(rank, name, count)
            flask_return.append(res)
        return flask_return


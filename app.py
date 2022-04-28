#  app.py
#  Code is working with Resume,JD and Search.
#  working with upload part
#  Working with C++ keyword
 



import time
import_time=time.time()
import nltk,re
from nltk.corpus import stopwords
# from sklearn.metrics.pairwise import cosine_similarity
import textract
# from nltk import sent_tokenize
from nltk.tokenize import word_tokenize

import os
# import sys
# import logging
# import six
# import string
# import pdfminer.settings
# pdfminer.settings.STRICT = False
# import pdfminer.high_level
# import pdfminer.layout
# from pdfminer.image import ImageWriter
import search


# In[3]:

# from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import glob
import warnings

# import requests
from flask import (Flask, render_template, request)
# from gensim.summarization import summarize
#from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from werkzeug.utils import secure_filename

import pdf2txt as pdf

print("\n Import packages time: ", time.time()- import_time)

warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

app = Flask(__name__)




app.config['UPLOAD_FOLDER'] = 'Original_Resumes/'
app.config['ALLOWED_EXTENSIONS'] = set(['txt', 'pdf'])


class jd:
    def __init__(self, name):
        self.name = name

class ResultElement:
    def __init__(self, rank, filename, similarity):
        self.rank = rank
        self.filename = filename
        self.similarity = similarity
    # def printresult(self):
    #     print(str("Rank " + str(self.rank+1) + " :\t " + str(self.filename)))


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']


def getfilepath(loc):
    # print("loc-->",loc)
    temp = str(loc).split('\\')
    return temp[-1]

def getdocxfilepath(loc):
    temp = str(loc).split('\\')
    # print("temp: ",temp)
    return temp[-1][:-5]

def deletefiles():
    print("s")




def extractTextFromDocx(filename):
    a = textract.process(filename)
    a = a.replace(b'\n',  b' ')
    a = a.replace(b'\r',  b' ')
    b = str(a)
    return b



def docx_pfd_cleaner(text):

    text = text.replace("/"," ").replace('+','qwe')
    email = re.findall(r"[a-z0-9\.\-+_]+@[a-z0-9\.\-+_]+\.[a-z]+", text)
    phone_no = re.findall(r'(\d{3}[-\.\s]??\d{3}[-\.\s]??\d{4}|\(\d{3}\)\s*\d{3}[-\.\s]??\d{4}|\d{3}[-\.\s]??\d{4})', text)

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
#             print(word)
            if word.lower() not in stopwords.words('english'):
                temp.append(word.lower())
            
        if temp != []:
            final_temp.append(temp)

    resume_text = []
    for sen in final_temp:
        resume_text.append(" ".join(sen).replace('qwe','+'))
    resume_text.append(" ".join(email))
    resume_text.append(" ".join(phone_no))
    
    

    return resume_text

@app.route('/', methods =['GET', 'POST'])
def index():
#    return render_template('index.html')
    x = []
    print("current dir--->",os.getcwd())
    # os.chdir('C:/Users/M1050766/Downloads/Resume_Recommendation-Final/resumizer-master')
    for file in glob.glob("./Job_Description/*.txt"):
        res = jd(file)
        # print("res-->",res)
        # path1 = jd(getfilepath(file))
        # print("file path-->",path1)
        x.append(jd(getfilepath(file)))
        # print("x:",x)
    return render_template('index.html', results = x)



@app.route('/uploadres', methods=['GET', 'POST'])
def uploadres():
    # os.chdir('C:/Users/M1050766/Downloads/Automated-Resume-Screening-System-master/Automated-Resume-Screening-System-master/resumizer-master')
    if request.method == 'POST':
        files = glob.glob('./Original_Resumes/*')
        for f in files:
            os.remove(f)

        uploaded_files = request.files.getlist("file[]")
      
        for file in uploaded_files:
            filename = secure_filename(file.filename)
            file.save(os.path.join('./Original_Resumes', filename))
        # return render_template('index.html')

    x = []
    os.chdir('C:/Users/M1050766/Downloads/Resume_Recommendation-Final/resumizer-master')
    for file in glob.glob("./Job_Description/*.txt"):
        res = jd(file)
        # print("res-->",res)
        # path1 = jd(getfilepath(file))
        # print("file path-->",path1)
        x.append(jd(getfilepath(file)))
     
    return render_template('index.html', results = x)


@app.route('/uploaddes', methods=['GET', 'POST'])
def uploaddes():
    if request.method == 'POST':
        files = glob.glob('./Job_Description/*')
        for f in files:
            os.remove(f)

        file = request.files['file']
        filename = secure_filename(file.filename)
        file.save(os.path.join('./Job_Description', filename))
        return render_template('index.html')
     
    return render_template('index.html')

########################################################################## Search String

@app.route('/resultsearch' ,methods = ['POST', 'GET'])
def resultsearch():
    if request.method == 'POST' or request.method == 'GET' :
        search_st = request.form.get('Name')
        print(search_st)
    result = search.res(search_st)
    # return result
    return render_template('search_result.html', results = result)






@app.route('/results')
def resume():
    Resume_Vector = []
    Ordered_list_Resume = []
    Ordered_list_Resume_Score = []
    LIST_OF_PDF_FILES = []
    LIST_OF_FILES_DOCX = []
    LIST_OF_FILES = []
    read_time=time.time()
    for file in glob.glob("./Original_Resumes/*.pdf"):
        LIST_OF_PDF_FILES.append(file)
    for file in glob.glob('./Original_Resumes/*.docx'):
        LIST_OF_FILES_DOCX.append(file)
    LIST_OF_FILES = 0
    LIST_OF_FILES = LIST_OF_FILES_DOCX + LIST_OF_PDF_FILES

    files = glob.glob('./Parsed_Resumes/*')
    for f in files:
    	os.remove(f)


    print("Total Files to Parse\t :" , len(LIST_OF_FILES))
    print("DOCX files to Parse\t :" , len(LIST_OF_FILES_DOCX))
    print("PDF Files to Parse\t :" , len(LIST_OF_PDF_FILES))
    
    print("####### PARSING ########")
    
    for i in LIST_OF_PDF_FILES:
        pdf.extract_text([str(i)] , all_texts=None , output_type='text' , outfile='Parsed_Resumes/'+getfilepath(i)+'.txt')

    for i in LIST_OF_FILES_DOCX:
        try:
            
            # docx_text = extractTextFromDocx(i)
            
            # # print("docxtxt------------------------->",docx_text)
            # cleaned_text = docx_pfd_cleaner(docx_text)
            # # print("cleaned_text------------------------->",cleaned_text)
            

            # with open('./Parsed_Resumes/'+getdocxfilepath(i)+'.pdf.txt', 'w') as f:
            #     f.write(". ".join(cleaned_text))


            docx_text = extractTextFromDocx(i)
            
            # # print("docxtxt------------------------->",docx_text)
            # cleaned_text = docx_pfd_cleaner(docx_text)
            # # print("cleaned_text------------------------->",cleaned_text)
            

            with open('./Parsed_Resumes/'+getdocxfilepath(i)+'.pdf.txt', 'w') as f:
                f.write(docx_text)

        except Exception as e: 
            print(e)

    # f.close()

    print("Done Parsing.")
    print("\n Reading  and writing time: ",(time.time()-read_time))


    cleaning_time=time.time()
    Job_Desc = 0
    LIST_OF_TXT_FILES = []
    for file in glob.glob("./Job_Description/*.txt"):
        # LIST_OF_TXT_FILES.append(file)

    # for i in LIST_OF_TXT_FILES:
        f = open(file , encoding="utf8")
        text = f.read()
        
        tttt = str(text)
        # print(tttt)
####################################################################

        cleaned_jd = docx_pfd_cleaner(tttt)
        # cleaned_jd = [cleaned_jd]
        cleaned_jd = " ".join(cleaned_jd)
        cleaned_jd = [cleaned_jd]
        print("JD-->",cleaned_jd)
        
        f.close()
        vectorizer = TfidfVectorizer(stop_words='english',ngram_range=(1,2),token_pattern = r"(?u)\b\w+\b")
        vectorizer.fit(cleaned_jd)
        vector = vectorizer.transform(cleaned_jd)

        print("\n\n Feature names--> ",vectorizer.get_feature_names())
        Features = vectorizer.get_feature_names()
        print(len(Features))
        # print(vector.shape)
        # print(vector.toarray())
        Job_Desc = vector.toarray()
        print("JD Vectors",Job_Desc)
        # print("\n\n")

    
    LIST_OF_TXT_FILES = []
    for file in glob.glob("./Parsed_Resumes/*.txt"):
        # LIST_OF_TXT_FILES.append(file)

    # for i in LIST_OF_TXT_FILES:
        Ordered_list_Resume.append(file)
        f = open(file , encoding="utf8")
        text = f.read()
        
        tttt = str(text)

        tttt = docx_pfd_cleaner(tttt)
        
        tttt = ' '.join(tttt)
        text = [tttt]
        # print(text)
        # print("\n\n")
        # print(text)
        f.close()

        vector = vectorizer.transform(text)
        # print(vector.shape)
        # print(vector.toarray())
        # aaa = vector.toarray()#########################
        # print("Resume Vector",aaa)
        # print("\n\n")
        Resume_Vector.append(vector.toarray())
        # print("\n\n")

    print("\nCleaning Time: ",(time.time()-cleaning_time))


    ranking_time=time.time()
    for i in Resume_Vector:
        # print("This is a single resume" , i)

        samples = i
        
        neigh = NearestNeighbors(n_neighbors=1, algorithm='auto', metric='cosine',leaf_size=30)
        neigh.fit(samples) 
 
        Ordered_list_Resume_Score.extend(neigh.kneighbors(Job_Desc)[0][0].tolist())
        
    flask_return = []
    
    # stoplist = stopwords.words('english')
    # stoplist.append('\n')
    
    # dir='textresume'
    # if os.path.exists("textresume"):
    #     print("Folder Exists")
    #     shutil.rmtree("textresume",  ignore_errors=True)
    #     print("Folder Deleted")

    # os.mkdir("textresume")
    # files_no_ext = [".".join(f.split(".")[:-1]) for f in os.listdir('Original_Resumes')]
    # # print(files_no_ext,Ordered_list_Resume_Score)
    # for i , j in sorted(zip(Ordered_list_Resume_Score, files_no_ext)):
    #     print(i,"-->", j)
    # # print("knn")
    # print(Ordered_list_Resume_Score)
    # for f in files_no_ext:
    #     a=open('textresume/'+f+'.txt','a')
    #     a.close()
    # resume_pdf=os.listdir('Original_Resumes')
    # resume_txt=os.listdir('textresume')


    # def extract_text(files=[], outfile=[],
    #         _py2_no_more_posargs=None,  # Bloody Python2 needs a shim
    #         no_laparams=False, all_texts=None, detect_vertical=None, # LAParams
    #         word_margin=None, char_margin=None, line_margin=None, boxes_flow=None, # LAParams
    #         output_type='text', codec='utf-8', strip_control=False,
    #         maxpages=0, page_numbers=None, password="", scale=1.0, rotation=0,
    #         layoutmode='normal', output_dir=None, debug=False,
    #         disable_caching=False, **other):
    #     if _py2_no_more_posargs is not None:
    #         raise ValueError("Too many positional arguments passed.")
    #     """if not files:
    #         raise ValueError("Must provide files to work upon!")"""

    #     # If any LAParams group arguments were passed, create an LAParams object and
    #     # populate with given args. Otherwise, set it to None.
    #     if not no_laparams:
    #         laparams = pdfminer.layout.LAParams()
    #         for param in ("all_texts", "detect_vertical", "word_margin", "char_margin", "line_margin", "boxes_flow"):
    #             paramv = locals().get(param, None)
    #             if paramv is not None:
    #                 setattr(laparams, param, paramv)
    #     else:
    #         laparams = None

    #     imagewriter = None
    #     if output_dir:
    #         imagewriter = ImageWriter(output_dir)

    #     """if output_type == "text" and outfile != "-":
    #         for override, alttype in (  (".htm", "html"),
    #                                 (".html", "html"),
    #                                 (".xml", "xml"),
    #                                 (".tag", "tag"),
    #                              (".txt","text")):
    #         if outfile.endswith(override):
    #             output_type = alttype"""

    #     if outfile == []:
    #         outfp = sys.stdout
    #         if outfp.encoding is not None:
    #             codec = 'utf-8'
    #     else:
    #         i=0
    #         for outfi in outfile:
    #             fname=files[i]
    #             i+=1
    #             outfp = open('textresume/'+outfi, "w", encoding='utf-8')

    #             Temp = fname.split(".")
    #             if Temp[1] == "pdf" or Temp[1] == "Pdf" or Temp[1] == "PDF":
    #                 with open('Original_Resumes/'+fname, "rb") as fp:
    #                     pdfminer.high_level.extract_text_to_fp(fp, outfp, codec='utf-8')

    #             if Temp[1] == "docx" or Temp[1] == "Docx" or Temp[1] == "DOCX":
                    
    #                 docx_text = extractTextFromDocx('Original_Resumes/'+fname)
            
    #                 cleaned_text = docx_pfd_cleaner(docx_text)
            
    #                 with open('./textresume/'+getdocxfilepath(fname)+'.txt', 'w',encoding='utf-8') as f:
    #                     f.write(". ".join(cleaned_text))
    #                     f.close()
    #             outfp.close()
    #     return

    
    Z = [(j,x) for j,x in sorted(zip(Ordered_list_Resume_Score,Ordered_list_Resume))]

    
    rank_list=[]
    for n,i in enumerate(Z):
        
        name = getfilepath(i[1])
        similarity=(1.00-i[0])*100
        similarity =round(similarity,2)
        name = name.split('.')[0]
        rank = n+1
        print(rank,"--->",similarity,"--->", name)
        rank_list.append(name)
        res = ResultElement(rank, name, similarity)
        # print(res)
        flask_return.append(res)
        # res.printresult()
        #print(f"Rank{res.rank+1} :\t {res.filename}")
    # print("rank_list--> ",rank_list)
    print("\nRanking time: ",(time.time() - ranking_time))
    with open('Rank_list.txt', 'w',encoding='utf-8') as f:
        count=1 
        for i in rank_list:
            # if count==13:
            #     break
            # else:
                f.write(str(count)+" "+str(i)+'\n')
                count+=1

    os.system("python WEE_json.py")
    return render_template('result.html', results = flask_return)

if __name__ == '__main__':
   # app.run(debug = True) 
    app.run('0.0.0.0' , 5000 , debug=True, threaded=True)


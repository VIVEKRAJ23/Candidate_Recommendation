#!/usr/bin/env python
# coding: utf-8

# In[1]:


import time
overall_time_taken = time.time()

from pathlib import Path
import nltk
import re
from nltk.corpus import stopwords
from nltk import sent_tokenize
from nltk.tokenize import word_tokenize
import os
import glob
import json
import spacy




# path to where all the original resume are stored.
path_to_original_resume = "./Original_Resumes/"

# path to where all the parsed resume are stored in txt format.
path_to_parsed = "./Parsed_Resumes/"

# path to Ranked_list.txt
path_to_ranked_list = "./Rank_list.txt"

# model path
model_path = "./Model/model_3_0_40/"

# path to where json file will be created
output_path = "./static"


list_of_original_files = glob.glob(path_to_original_resume+"/*")
list_of_parsed_files = glob.glob(path_to_parsed+"/*.txt")


# get the raking from rank_list
f = open(path_to_ranked_list, "r")
ranking = f.read().split("\n")
f.close()
ranking = [ele for ele in ranking if ele != ""]
ranked_list = [k.split(" ")[1] for k in ranking]
ranked_list_dict = {k.split(" ")[1]:k.split(" ")[0] for k in ranking}

# list of files in ranked in parsed folder
list_of_resumes = []
for file_name in ranked_list:
    for resume_name in list_of_parsed_files:
        if file_name in resume_name:
            list_of_resumes.append(resume_name)
      
path_to_name_map = {list_of_resumes[name]:ranked_list[name] for name in range(len(ranked_list))}


# text cleaner and splitting into sentences.
def cleaner(text):#,jd):
    #replacing special symbols / and ++ and +
    text = text.replace("/"," ").replace("++","shaktiman").replace("+","")
    # Sentence tokabizer, divide the sentences.
    sentences = nltk.sent_tokenize(text)
    # removing noises
    final_text = []
    for sen in sentences:
        tokens = word_tokenize(sen)
        words = [word for word in tokens if word.isalnum() or word.replace('.', '', 1).isdigit()]
        final_text.append(words)
    final_text = [ele for ele in final_text if ele != [] and len(ele) != 1]
    resume_text = []
    for sen in final_text:
        sentence = " ".join(sen)
        sentence = sentence.replace("shaktiman","++")
        resume_text.append(sentence.lower())
    return resume_text


# trimming the experience part from the sentence.
def exp_trimer(sen):
    sen = sen.split(" ")
    if len(sen) < 10:
#         print("Returning sen")
        return " ".join(sen)
    
    for word in range(len(sen)):
        if sen[word] in ["year","years"]:
            if word > 5 and word + 8 < len(sen):
#                 print("1. sentence:",sen[word-3:word+8])
                return " ".join(sen[word-3:word+8])
            elif word +5 > len(sen) and word - 5 > 0:
#                 print("2. sentence:",sen[word-5:])
                return " ".join(sen[word-5:])
            elif word < 10 and len(sen) > 11:
                if word < 4 : 
#                     print("3. sentence:",sen[: word + 4])
                    return " ".join(sen[: word + 4])
                else:
#                     print("4. sentence:",sen[word-3: word + 4])
                    return " ".join(sen[word-3: word + 4])
            else:
#                 print("5. sentence:",sen[word - 4: word + 3])
                return " ".join(sen[word - 4: word + 3])

# return sentence segment
def previous_5_words(sen,matched_char):
    sen = re.sub(' +', ' ', sen)
    sen = sen.replace(",","")
    matched_char = matched_char.replace(" ","")
    out = []
    sen = sen.split(" ")
    for word in range(len(sen)):
        if matched_char == sen[word] or matched_char in sen[word]:
            if len(sen) < 12:
                return " ".join(sen)
            elif word > 7 and word < len(sen)-7:
                return " ".join(sen[word-6:word+7])
            elif word == len(sen):
                return " ".join(sen[word-5:word+1])
            elif word < 7 and len(sen) > 10:
                return " ".join(sen[:word+7])
            elif word < 7 and len(sen) <= 7:
                return " ".join(sen[:word + 1])
            elif word < len(sen) and len(sen) >= word + 4 and word > 3 and word - 4 > 0:
                return " ".join(sen[word - 4:word + 5])
            else:
                return " ".join(sen)

#returns cleaned list of sentences
def read_file(path):
    f = open(path , encoding="utf8")
    text = f.read() 
    f.close()
    text = str(text)
    cleaned_text = cleaner(text)
    email_id_list = re.findall('\S+@\S+', text)
    numbers = re.findall(r'(\d{3}[-\.\s]??\d{3}[-\.\s]??\d{4}|\(\d{3}\)\s*\d{3}[-\.\s]??\d{4}|\d{3}[-\.\s]??\d{4})', text)
    phone_numbers = [ num for num in numbers if len(num.replace(" ","")) >= 9]
    return cleaned_text, email_id_list, phone_numbers


def education_checker(sen):
    edu_ch = ["college","university","education","institute"]
    for word in edu_ch:
        if word in sen:
            return True, word
    return False, "Null"

def company_checker(sen):
    company_ch = [" ltd "," llp "," pvt "," ltd.", "private",
                  " limited"," inc "," corporation"," inc."," company",
                  "technology solution"," pvt.","ltd"] #,"enterprise"
    for word in company_ch:
        if word in sen:
            return True, word
    return False, "Null"


print("\nLoading spaCy model...")
now = time.time()
nlp_updated = spacy.load(model_path)

def ORG_label(sen):
    ORG = []
    doc = nlp_updated(sen)
    for ent in doc.ents:
        if ent.label_ == "ORG":
            ORG.append(ent.text)
    return ORG
print("Model loading time:",time.time()-now)


# In[55]:


result = {name:{"exp":[],"education":[],"company":[]} for name in ranked_list}   

#Reading all the text
print("Extracting tags from resumes:")
now = time.time()
for resume_path in list_of_resumes:
    r_name = path_to_name_map[resume_path]
    resume,email_id_list,numbers = read_file(resume_path)
    for sentence in resume:
        if "year" in sentence:
            result[r_name]["exp"].append(exp_trimer(sentence))
        edu_check, word_matched = education_checker(sentence)
        if edu_check:
            trimmed = previous_5_words(sentence,word_matched)
            for model_output in ORG_label(trimmed):
                if model_output != []:
                    result[r_name]["education"].append(model_output)
                    # uncumment this to add trimmed text if model output is enpty
#                 else:
#                     result[r_name]["education"].append(trimmed)
        
        com_check, word_matched = company_checker(sentence)
        if com_check:
            trimmed = previous_5_words(sentence,word_matched)
            for model_output in ORG_label(trimmed):
                if model_output != []:
                    result[r_name]["company"].append(model_output)
                    # uncumment this to add trimmed text if model output is enpty
#                 else:
#                     result[r_name]["company"].append(trimmed)
            
        result[r_name]["email"] = email_id_list
        result[r_name]["mobile_number"] = numbers
        result[r_name]["rank"] = ranked_list_dict[r_name] 
print("Time taken to extract tags: ", time.time()-now)


print("\nSaving the output in json format...")

# Save the output in json format
with open(output_path+"/Result.json", "w") as outfile:
    json.dump(result, outfile)

print("\nTime taken for Entity extraction: ",time.time() - overall_time_taken, "sec")
print("Final json:")
print(result)

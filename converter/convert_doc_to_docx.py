import os
import win32com
import glob
from win32com import client as wc

def convert_doc_to_docx(resume):
    file_name = resume.split("/")[-1]
    file_name = file_name.split(".")[0]
    file_name = file_name.replace(" ","_")
    print(file_name)
    print(resume)
    
    #word.visible = 0
    in_file = os.path.abspath(resume)
    print("in_file",in_file)
    wb = word.Documents.Open(in_file)
    #out_file = "./converted/"+file_name+".docx"
    out_file = in_file.replace(".doc",".docx").replace("to_convert","converted")
    print(out_file)
    wb.SaveAs2(out_file, FileFormat=16) # file format for docx
    wb.Close()
    
    return 0

word = win32com.client.Dispatch("Word.Application")
path_selected = "./to_convert/"
selected_list = [file.replace("\\","/") for file in glob.iglob("./to_convert/*.doc")]

print("selected_list: ",selected_list)
for res in selected_list:
    print("____________________")
    print("file path", res)
    convert_doc_to_docx(res)
    #break
    #if ".doc" in res and ".docx" not in res:
    #convert_doc_to_docx(res)
word.Quit()



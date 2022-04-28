import glob
import os
import win32com.client

word = win32com.client.Dispatch("Word.Application")
word.visible = 0

for i, doc in enumerate(glob.iglob("*.doc")):
    in_file = os.path.abspath(doc)
    wb = word.Documents.Open(in_file)
    out_file = os.path.abspath("out{}.docx".format(i))
    wb.SaveAs2(out_file, FileFormat=16) # file format for docx
    wb.Close()

word.Quit()

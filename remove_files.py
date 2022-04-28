import os
import glob

list_of_files = glob.glob("./Ranked_resumes/*.txt")

print("removing Ranked files")

for file in list_of_files:
    os.remove(file)

list_of_files = glob.glob("./Parsed_Resumes/*.txt")

print("removing Parsed files")
for file in list_of_files:
    os.remove(file)



os.remove("./Rank_list.txt")

os.remove("./static/Result.json")

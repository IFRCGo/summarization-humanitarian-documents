import sys
sys.path.append("/idiap/temp/jbello/others/")
from utils import *
import json
from langdetect import detect

data_ref_directory = "/idiap/temp/jbello/data/preprocessed/sent3/aspects/"

data_file = 'language.sent3.json'


print('Partitioning data in english, spanish, french, arab and other languages!')
lang_part, _ = identify_language(data_ref_directory,'summary.')
print('Saving partition')
with open(os.path.join(data_ref_directory,data_file), 'w', encoding='utf8') as outfile:
    json.dump(lang_part, outfile)
print('Saved index partition!')

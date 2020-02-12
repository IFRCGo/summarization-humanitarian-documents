import sys
sys.path.append("/idiap/temp/jbello/others/")
from utils import *
from random import shuffle
import json

data_ref_directory = "/idiap/temp/jbello/data/preprocessed/sent3/"

language_file = "/idiap/temp/jbello/data/preprocessed/sent3/language.sent3.json"

data_file = 'partition.sent3.json'

def partition(data_directory, name_text, language_partition = None, language = 'en', train_prop = 0.7, val_prop = 0.1, test_prop = 0.2):
    if language_partition == None:
        '''Partition 70.10.20 by default'''
        fn, doc = read_directory(data_directory, name_text)
        ids = extract_id(fn)
    else:
        with open(language_partition) as json_file:
            ids = json.load(json_file)
        if language == 'en':
            ids = ids[0]
        elif language == 'es':
            ids = ids[1]
        elif language == 'fr':
            ids = ids[2]
        elif language == 'ar':
            ids = ids[3]
        else:
            print('not partition find for such a language!')
    #randomize order
    shuffle(ids)
    #make partition
    train_limit = int(len(ids)*train_prop)
    val_limit = train_limit + (int(len(ids)*val_prop))
    test_limit = len(ids)
    seq = [train_limit,val_limit,test_limit]
    result = []
    for i in range(0,len(seq)):
        chunk = []
        if (i == 0):
            for j in range(0, train_limit):
                chunk.append(ids[j])
        else:
            for j in range (seq[i-1], seq[i]):
                chunk.append(ids[j])
        result.append(chunk)
    return result

print('Partitioning data in train (70%), validation (10%) and test (20%)!')
idx_part = partition(data_ref_directory, name_text = 'summary.', language_partition = language_file)
print('Saving partition')
with open(os.path.join(data_ref_directory,data_file), 'w', encoding='utf8') as outfile:
    json.dump(idx_part, outfile)
print('Saved index partition!')

import sys
sys.path.append("/idiap/temp/jbello/others/")
from utils import *
from random import shuffle
import json

data_ref_directory = "/idiap/temp/jbello/data/preprocessed/sent3/aspects/"

language_file = "/idiap/temp/jbello/data/preprocessed/sent3/aspects/language.sent3.json"

aspects_file = '/idiap/temp/jbello/data/preprocessed/sent3/aspects/aspects.partition.json'

data_file = 'partition.sent3.json'

def partition(data_directory, name_text, language_partition = None, language = 'en', aspects_partition = None,  train_prop = 0.7, val_prop = 0.1, test_prop = 0.2):
    if language_partition == None:
        '''Partition 70.10.20 by default'''
        fn, doc = read_directory(data_directory, name_text)
        ids = [sub.replace(name_text,'') for sub in fn]
        ids = [sub.replace('.txt','') for sub in ids]
    else:
        with open(language_partition) as json_file:
            lang_ids = json.load(json_file)
        if aspects_partition != None:
            with open(aspects_partition) as json_file:
                asp_ids = json.load(json_file)
            #save for each language, just the text with aspects in filter 
            lang_ids[0] = list(set(asp_ids[0]) and set(lang_ids[0]))
            lang_ids[1] = list(set(asp_ids[0]) and set(lang_ids[1]))
            lang_ids[2] = list(set(asp_ids[0]) and set(lang_ids[2]))
            lang_ids[3] = list(set(asp_ids[0]) and set(lang_ids[3]))
        ids = lang_ids
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
idx_part = partition(data_ref_directory, name_text = 'summary.', language_partition = language_file, aspects_partition = aspects_file)
print('Saving partition')
with open(os.path.join(data_ref_directory,data_file), 'w', encoding='utf8') as outfile:
    json.dump(idx_part, outfile)
print('Saved index partition!')

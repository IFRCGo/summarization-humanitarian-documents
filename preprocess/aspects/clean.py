import os
import codecs
import json
import sys
sys.path.append('/idiap/temp/jbello/others/')
from utils import save_texts

data_directory = '/idiap/temp/jbello/data/original/aspects/'
DEEP_name_document = 'simplified_text'
DEEP_name_summary = 'tagged_excerpts'
DEEP_name_idx = 'lead_id'
DEEP_name_aspect = 'sectors'
save_path = '/idiap/temp/jbello/data/cleaned/aspects/'

class Errors:
    too_short_documents = 0
    empty_summaries = 0

min_document_tokens = 50

def read_directory(directory,name_document, name_summary, name_idx):
    '''Read json documents in the directory, containing documents, 
    their summaries, and their specific identifier'''
    dict_obs = {}
    for data_file in os.listdir(directory):
        if ".json" in data_file:
            with codecs.open(os.path.join(directory,data_file),encoding="utf8") as f:
                data = json.load(f)
                for item in data:
                    if (name_document and name_summary and name_idx in item):
                        temp = item.get(name_summary)
                        for entry_item in temp:
                            if (type(entry_item) == dict):
                                temp_sec = entry_item
                            elif(type(entry_item) == list and len(entry_item) > 0):
                                temp_sec = entry_item[0]
                            for sec_item in temp_sec.get('sectors'):
                                #if the document hasn't been ingested
                                if item.get(name_idx) not in dict_obs:
                                    dict_obs[item.get(name_idx)] = {}
                                    dict_obs[item.get(name_idx)]['documents'] = item.get(name_document)
                                    dict_obs[item.get(name_idx)]['summaries'] = temp_sec.get('excerpt')
                                    dict_obs[item.get(name_idx)]['aspects'] = sec_item
                                #if the document was already ingested
                                else: 
                                    if dict_obs[item.get(name_idx)]['aspects'] == sec_item:
                                        #verify case where there is already an entry for such document in the same aspect
                                        dict_obs[item.get(name_idx)]['summaries'] = dict_obs[item.get(name_idx)]['summaries'] + ' ' + temp_sec.get('excerpt')
                                    else:
                                        #if the document has already be ingested, but for another aspect, we create a new observation
                                        dict_obs[item.get(name_idx)] = {}
                                        dict_obs[item.get(name_idx)]['documents'] = item.get(name_document)
                                        dict_obs[item.get(name_idx)]['summaries'] = temp_sec.get('excerpt')
                                        dict_obs[item.get(name_idx)]['aspects'] = sec_item
        documents = []
        summaries = []
        idx = []
        idx_sum = []
        for item in dict_obs:
    	    documents.append(dict_obs[item].get('documents'))
    	    summaries.append(dict_obs[item].get('summaries'))
    	    idx.append(item)
    	    idx_sum.append(str(dict_obs[item].get('aspects'))+ '.' +str(item))
        idx_sum = [sub.replace('/', '_') for sub in idx_sum]
    return documents, summaries, idx, idx_sum

def remove_boolean_and_empty_docs(idx, idx_sum, documents,summaries):
    idx_int = []
    idx_sum_int = []
    documents_int = []
    summaries_int = []
    for item in range(0,len(documents)):
        if (type(documents[item]) is not bool and documents[item].isspace() == False):
            idx_int.append(idx[item])
            idx_sum_int.append(idx_sum[item])
            documents_int.append(documents[item])
            summaries_int.append(summaries[item])
    return idx_int, idx_sum_int, documents_int, summaries_int

def remove_short_documents(idx, idx_sum, documents,summaries):
    idx_int = []
    idx_sum_int = []
    documents_int = []
    summaries_int = []
    for item in range(0, len(documents)):
        # Restrict minimum length of document
        if len(documents[item].split(" ")) < min_document_tokens:
            Errors.too_short_documents += 1
        else:
            idx_int.append(idx[item])
            idx_sum_int.append(idx_sum[item])
            documents_int.append(documents[item])
            summaries_int.append(summaries[item])
    return idx_int, idx_sum_int, documents_int, summaries_int

def process_data(directory, name_document, name_summary, name_idx):
    print("Processing directory: %s" % directory)
    documents, summaries, idx, idx_sum = read_directory(directory, name_document, name_summary, name_idx)
    idx, idx_sum, documents,summaries =  remove_boolean_and_empty_docs(idx, idx_sum, documents,summaries)
    print("Boolean and empty documents removed")
    idx, idx_sum, documents,summaries =  remove_short_documents(idx, idx_sum, documents,summaries)
    print("Short documents removed")   
    return idx, idx_sum, documents, summaries

print('begin saving texts')
idx,idx_sum,documents, summaries = process_data(data_directory,DEEP_name_document, DEEP_name_summary, DEEP_name_idx)
save_texts(save_path, 'document.',documents,idx)
save_texts(save_path, 'summary.',summaries,idx_sum)
print('saved texts')

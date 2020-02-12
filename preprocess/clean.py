import os
import codecs
import json
import re


min_document_tokens = 50

data_directory = "/idiap/temp/jbello/data/original/"

relative_save_path = "/idiap/temp/jbello/data/cleaned/"


class Errors:
    too_short_documents = 0
    empty_summaries = 0

def read_directory(directory,name_document, name_summary, name_idx):
    '''Read json documents in the directory, containing documents, 
    their summaries, and their specific identifier'''
    idx = []
    documents= []
    summaries = []
    for data_file in os.listdir(directory):
        if ".json" in data_file:
            with codecs.open(os.path.join(directory,data_file),encoding="utf8") as f:
                data = json.load(f)
                for item in data:
                    if (name_document and name_summary and name_idx in item):
                        idx.append(item.get(name_idx))
                        documents.append(item.get(name_document))
                        summaries.append(item.get(name_summary))
    summaries = join_entries(summaries)
    return idx, documents, summaries            

def join_entries(row):
    '''Convert from a list of lists, to a list of lemmatized text.
    Concatenate the text in the same inner list.'''
    summaries = []
    for i in range(0,len(row)):
        if (i%1000 == 0):
            print("Processing summary %i of %i; %.2f percent done" %
                  (i, len(row), float(i) * 100.0 / float(len(row))))
        text = ''
        if (i > 0):
            for j in range(0,len(row[i])):
                text = text + row[i][j].get('excerpt') + ' '
        summaries.append(text)
    return summaries

def process_data(directory,name_document, name_summary, name_idx):
    print("Processing directory: %s" % directory)
    idx, documents,summaries = read_directory(directory,name_document, name_summary, name_idx)
    num_documents = len(documents)
    idx, documents,summaries =  remove_boolean_and_empty_docs(idx, documents,summaries)
    print("Boolean and empty documents removed")
    idx, documents,summaries =  remove_short_documents(idx, documents,summaries)
    print("Short documents removed")   
    print("--Too short documents: %d" % Errors.too_short_documents)
    return idx, documents, summaries


def remove_boolean_and_empty_docs(idx, documents,summaries):
    idx_int = []
    documents_int = []
    summaries_int = []
    for item in range(0,len(documents)):
        if (type(documents[item]) is not bool and documents[item].isspace() == False):
            idx_int.append(idx[item])
            documents_int.append(documents[item])
            summaries_int.append(summaries[item])
    return idx_int, documents_int, summaries_int


def remove_short_documents(idx, documents,summaries):
    idx_int = []
    documents_int = []
    summaries_int = []
    for item in range(0, len(documents)):
        # Restrict minimum length of document
        if len(documents[item].split(" ")) < min_document_tokens:
            Errors.too_short_documents += 1
        else:
            idx_int.append(idx[item])
            documents_int.append(documents[item])
            summaries_int.append(summaries[item])
    return idx_int, documents_int, summaries_int


def save_documents(idx, documents, summaries, relative_path):
    for i in range(0,len(documents)):
        complete_doc_name = os.path.join(relative_path,'document.'+ str(idx[i]) +'.txt')
        with open(complete_doc_name,'w', encoding = 'utf-8') as f:
            f.write(documents[i])
            f.write("\n")
    #save non-empty summaries
    for i in range(0,len(summaries)):
        if summaries[i].isspace() == False:
            complete_sum_name = os.path.join(relative_path,'summary.'+ str(idx[i]) +'.txt')
            with open(complete_sum_name,'w', encoding = 'utf-8') as f:
                f.write(summaries[i])
                f.write("\n")
        else:
            Errors.empty_summaries += 1

            
print("start processing")
idx, documents, summaries = process_data(data_directory,'simplified_text','tagged_excerpts','lead_id')
print("start saving")
save_documents(idx, documents, summaries, relative_save_path)
print("Number of saved documents: %d" % len(documents))
print("Number of saved summaries: %d" % (len(summaries) - Errors.empty_summaries))
print("DONE")

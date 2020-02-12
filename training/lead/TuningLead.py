import sys
import codecs
sys.path.append("/idiap/temp/jbello/models/lead/")
from Lead import summarize
sys.path.append("/idiap/temp/jbello/others/")
from utils import *
import os
import subprocess


data_directory = "/idiap/temp/jbello/data/preprocessed/sent3/"

partition_file = "/idiap/temp/jbello/data/preprocessed/sent3/partition.sent3.json"

data_ref_directory = "/idiap/temp/jbello/data/validation/reference/"

save_path = "/idiap/temp/jbello/data/validation/lead/"

config_file_path = "lead_config.xml"

rouge_script_path = "/idiap/temp/jbello/pyrouge/ROUGE-1.5.5.pl"

data_folder_rouge = "/idiap/temp/jbello/pyrouge/data/"

sentences = [3,4,5,6]

def train_lead(directory,partition_file, number_sentences):
    filename, docs = select_partition(directory,partition_file, partition = 'validation', name_text = 'document.')
    summaries = []
    for i in range(0, len(docs)):
        if (i%1000 == 0):
            print(str(i), ' out of ',str(len(docs)), ' documents summarized.')
        try:
            summaries.append(summarize(docs[i], nb_sentences = number_sentences))
        except:
            print('Error at observation ', str(i), '. Complete document returned.')
            summaries.append(docs[i])
    temp_id = extract_id(filename)     
    return temp_id, summaries

def tune_hyperparameter_lead (directory, partition_file, sentences = [3]):
    for sentence in sentences:
        temp_id,summaries = train_lead(directory,partition_file, sentence)
        relative = 'lead'+str(sentence)+'/'
        relative_save_path = os.path.join(save_path,relative)
        if not os.path.exists(relative_save_path):
            os.makedirs(relative_save_path)
        save_texts(relative_save_path, 'summary.', summaries, temp_id) 
        print('converting each text in rouge format:')
        write_text_with_rouge_format(relative_save_path, data_ref_directory)
        relative_config_file_path = os.path.join(relative_save_path,config_file_path)
        write_config_static(relative_save_path, 'cand.(\d+).txt',
                    data_ref_directory, 'ref.(\d+).txt',
                    relative_config_file_path, system_id='1')
        print('Saved configuration!')
    print('Grid search on different lengths of summary finished!')   
    return

print('start tuning of hyperparameters:')
tune_hyperparameter_lead(data_directory, partition_file, sentences)
print('saved configuration for rouge evaluation')

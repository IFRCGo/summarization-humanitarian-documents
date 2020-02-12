import sys
import codecs
sys.path.append("/idiap/temp/jbello/models/text_rank_bm25/")
from TextRankBM25 import summarize
sys.path.append("/idiap/temp/jbello/others/")
from utils import *
import os
import subprocess


data_directory = "/idiap/temp/jbello/data/preprocessed/sent3/"

partition_file = "/idiap/temp/jbello/data/preprocessed/sent3/partition.sent3.json"

data_ref_directory = "/idiap/temp/jbello/data/validation/reference/"

save_path = "/idiap/temp/jbello/data/validation/text_rank_bm25/"

config_file_path = "text_rank_bm25_config.xml"

rouge_script = '/idiap/temp/jbello/others/pyrouge/ROUGE-1.5.5.pl'

rouge_data_folder = '/idiap/temp/jbello/others/pyrouge/data'


ratios = [0.1,0.2,0.3]
word_count = [300,500,1000]
weight_th = [1.e-3,1.e-1]

def train_text_rank_bm_25(directory,partition_file, ratio = 0.2, word_count=None,weight_threshold = 1.e-3):
    filename, docs = select_partition(directory,partition_file, partition = 'validation', name_text = 'document.')
    summaries = []
    for i in range(0, len(docs)):
        if (i%1000 == 0):
            print(str(i), ' out of ',str(len(docs)), ' documents summarized.')
        try:
            summaries.append(summarize(docs[i], ratio, word_count, weight_threshold))
        except:
            print('Error at observation ', str(i), '. Complete document returned.')
            summaries.append(docs[i])
    temp_id = extract_id(filename)  
    print(temp_id)   
    return temp_id, summaries

def tune_hyperparameter_text_rank_bm25 (directory, partition_file, ratios = [0.2], word_count = [None],weight_thershold = [1.e-3]):
    for ratio in ratios:
        for weight in weight_th:
            temp_id,summaries = train_text_rank_bm_25(directory,partition_file, ratio = ratio, weight_threshold=weight)
            relative = str(ratio)+'_'+str(weight)+'/'
            relative_save_path = os.path.join(save_path,relative)
            if not os.path.exists(relative_save_path):
                os.makedirs(relative_save_path)
            save_texts(relative_save_path, 'summary.', summaries, temp_id)
            print('converting each text in rouge format:')
            write_text_with_rouge_format(relative_save_path, data_ref_directory)
            print('writing rouge configuration file:')
            relative_config_file_path = os.path.join(relative_save_path,config_file_path)
            print(relative_config_file_path)
            write_config_static(relative_save_path, 'cand.(\d+).txt',
                        data_ref_directory, 'ref.(\d+).txt',
                        relative_config_file_path, system_id='1')
            print('Saved configuration!')
    print('Grid search on ratios and weight thresholds finished!')

    for word_c in word_count:
        for weight in weight_th:
            temp_id,summaries = train_text_rank_bm_25(directory,partition_file, word_count = word_c, weight_threshold=weight)
            relative = str(word_c)+'_'+str(weight)+'/'
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
    print('Grid search on maximum word count and weight thresholds finished!')   
    return


print('start tuning of hyperparameters:')
tune_hyperparameter_text_rank_bm25(data_directory, partition_file, ratios, word_count,weight_th)
print('Saved reports of hyperparameters performance.')

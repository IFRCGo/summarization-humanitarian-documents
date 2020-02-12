import sys
sys.path.append("/idiap/temp/jbello/others/")
from utils import *
sys.path.append("/idiap/temp/jbello/models/lead/")
from Lead import summarize
import subprocess

data_directory = "/idiap/temp/jbello/data/preprocessed/sent3/"

data_ref_directory = "/idiap/temp/jbello/data/results/reference/"

data_hyp_directory = "/idiap/temp/jbello/data/results/lead/"

partition_file = "/idiap/temp/jbello/data/preprocessed/sent3/partition.sent3.json"

config_file_path = "/idiap/temp/jbello/data/results/lead/test_lead_config.xml"

rouge_script = '/idiap/temp/jbello/pyrouge/ROUGE-1.5.5.pl'

rouge_data_folder = '/idiap/temp/jbello/pyrouge/data'


def test_lead(directory, partition_file):
    filename, docs = select_partition(directory,partition_file, partition = 'test', name_text = 'document.')
    summaries = []
    for i in range(0, len(docs)):
        if (i%1000 == 0):
            print(str(i), ' out of ',str(len(docs)), ' documents summarized.')
        summaries.append(summarize(docs[i],nb_sentences = 4)) 
    temp_id = extract_id(filename)    
    return temp_id, summaries

temp_id, summaries = test_lead(data_directory, partition_file)
save_texts(data_hyp_directory, 'summary.', summaries, temp_id)
print('Saved summaries: ',len(temp_id))
print('converting each text in rouge format:')
write_text_with_rouge_format(data_hyp_directory, data_ref_directory)
print('writing rouge configuration file:')
write_config_static(data_hyp_directory, 'cand.(\d+).txt',
                        data_ref_directory, 'ref.(\d+).txt',
                        config_file_path, system_id='1')
print('Saved configuration!')

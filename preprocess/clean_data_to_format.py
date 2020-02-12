import sys
sys.path.append('/idiap/temp/jbello/others')
from utils import join_texts, save_texts, select_partition
sys.path.append('/idiap/temp/jbello/preprocess/')
from textcleaner import *
from preprocessing import *

data_directory = '/idiap/temp/jbello/data/preprocessed/sent3/'

partition_file = '/idiap/temp/jbello/data/preprocessed/sent3/partition.sent3.json'

def clean_data_to_format(directory,partition, part, name_text, save_name):
    print ('Begin reading of data')
    _, documents = select_partition(directory, partition, part, name_text)
    print('Begin preprocessing of data')
    FILTERS = [
        lambda x: x.lower(), strip_tags, strip_punctuation, strip_multiple_whitespaces, strip_numeric, strip_short        
    ]
    output = ''
    for document in documents:
        original_sentences = split_sentences(document)
        filtered_sentences = [join_words(sentence) for sentence in preprocess_documents(original_sentences, FILTERS)]
        filtered_sentences = ' ###SENT### '.join(filtered_sentences)
        output = output + '\"'+filtered_sentences+'\" \n'
    print('Saving data')
    save_texts(data_directory, save_name,[output],[''])
    print('Saved data')

clean_data_to_format(data_directory, partition_file, 'training', 'document.','train.src.')
clean_data_to_format(data_directory, partition_file, 'training', 'summary.','train.tgt.')

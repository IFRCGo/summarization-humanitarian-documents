import sys
sys.path.append("/idiap/temp/jbello/others/")
from utils import *
import json

data_ref_directory = "/idiap/temp/jbello/data/preprocessed/sent3/"

partition_file = "/idiap/temp/jbello/data/preprocessed/sent3/partition.sent3.json"

save_path = "/idiap/temp/jbello/data/validation/reference/"


print('Begin selection of reference summaries in the validation partition:')
filename, docum = select_partition(data_ref_directory,partition_file,'validation','summary.')
print ('Saving summaries in new path:')
save_texts(save_path,'summary.', docum, extract_id(filename))
print('Saved documents!')

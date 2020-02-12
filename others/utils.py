import numpy as np
import scipy
import scipy.stats
#import matplotlib
#import matplotlib.pyplot as plt
#from rouge import Rouge
#from pyrouge import Rouge155
import os
import codecs
import re
import sys
import json
from langdetect import detect

def read_directory(directory, name_text = '.txt'):
    '''Read documents in directory. The type of text to read can be precised, 
    given the name of the text (whether is document/summary)'''
    filenames = []
    documents = []
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if (name_text) in filename:
            with codecs.open(os.path.join(directory,filename),encoding="utf8") as f:
                filenames.append(filename)
                documents.append(f.read())
    return filenames, documents


def save_texts(relative_path, name_text, texts, id_texts):
    for i in range(0,len(texts)):
        complete_text_name = os.path.join(relative_path,name_text+ str(id_texts[i]) +'.txt')
        with open(complete_text_name,'w', encoding = 'utf-8') as f:
            f.write(texts[i])
            
def extract_id(string_with_id):
    if type(string_with_id) == list:
        id_doc = []
        for i in range (0,len(string_with_id)):
            id_doc.append(re.findall(r'\d+',string_with_id[i])[0])
        return id_doc
    elif(type(string_with_id) == str):
        return re.findall(r'\d+',string_with_id)[0]

def select_labeled_docs(directory):
    '''Return list of documents, that have associated summaries, and the list of their names'''
    #identify available summaries
    names_ground, _ = read_directory(directory, name_text = 'summary.')
    temp_id = extract_id(names_ground)
    #read all documents
    filenames, docs = read_directory(directory, name_text = 'document.')
    id_doc = extract_id(filenames)
    #filter documents with summaries
    labeled_docs = []
    for i in range(0,len(temp_id)):
        name_t = "document."+str(temp_id[i])+".txt"
        if name_t in filenames:
            labeled_docs.append(docs[id_doc.index(temp_id[i])])
    return temp_id, labeled_docs

#taken from GitHub @bheinzerling
def write_config_static(system_dir, system_filename_pattern,
                        model_dir, model_filename_pattern,
                        config_file_path, system_id=None):
    """
    Write the ROUGE configuration file, which is basically a list
    of system summary files and their corresponding model summary
    files.
    pyrouge uses regular expressions to automatically find the
    matching model summary files for a given system summary file
    (cf. docstrings for system_filename_pattern and
    model_filename_pattern).
        system_dir:                 Path of directory containing
                                    system summaries.
        system_filename_pattern:    Regex string for matching
                                    system summary filenames.
        model_dir:                  Path of directory containing
                                    model summaries.
        model_filename_pattern:     Regex string for matching model
                                    summary filenames.
        config_file_path:           Path of the configuration file.
        system_id:                  Optional system ID string which
                                        will appear in the ROUGE output.
    """
    
    system_filenames = [f for f in os.listdir(system_dir)]
    system_models_tuples = []

    system_filename_pattern = re.compile(system_filename_pattern)
    for system_filename in sorted(system_filenames):
        match = system_filename_pattern.match(system_filename)
        if match:
            id = match.groups(0)[0]
            model_filenames = get_model_filenames_for_id(
                id, model_dir, model_filename_pattern)
            system_models_tuples.append(
                (system_filename, model_filenames))
    if not system_models_tuples:
        raise Exception(
            "Did not find any files matching the pattern {} "
            "in the system summaries directory {}.".format(
                system_filename_pattern.pattern, system_dir))
    

    with codecs.open(config_file_path, 'w', encoding='utf-8') as f:
        f.write('<ROUGE-EVAL version="1.55">')
        for task_id, (system_filename, model_filenames) in enumerate(
                system_models_tuples, start=1):
            eval_string = get_eval_string(
                task_id, system_id,
                system_dir, system_filename,
                model_dir, model_filenames)
            f.write(eval_string)
        f.write("</ROUGE-EVAL>")
        

#taken from GitHub @bheinzerling
def get_eval_string(
        task_id, system_id,
        system_dir, system_filename,
        model_dir, model_filename):
    """
    ROUGE can evaluate several system summaries for a given text
    against several model summaries, i.e. there is an m-to-n
    relation between system and model summaries. The system
    summaries are listed in the <PEERS> tag and the model summaries
    in the <MODELS> tag. pyrouge currently only supports one system
    summary per text, i.e. it assumes a 1-to-n relation between
    system and model summaries.
    """
    peer_elems = "<P ID=\"{id}\">{name}</P>".format(
        id=system_id, name=system_filename)

    model_elems = "<M ID=\"{id}\">{name}</M>".format(
        id='A', name=model_filename)

    eval_string = """
    <EVAL ID="{task_id}">
        <MODEL-ROOT>{model_root}</MODEL-ROOT>
        <PEER-ROOT>{peer_root}</PEER-ROOT>
        <INPUT-FORMAT TYPE="SEE">
        </INPUT-FORMAT>
        <PEERS>
            {peer_elems}
        </PEERS>
        <MODELS>
            {model_elems}
        </MODELS>
    </EVAL>
""".format(
        task_id=task_id,
        model_root=model_dir, model_elems=model_elems,
        peer_root=system_dir, peer_elems=peer_elems)
    return eval_string

#taken from GitHub @bheinzerling 
def get_model_filenames_for_id(id, model_dir, model_filenames_pattern):
    pattern = re.compile(model_filenames_pattern.replace('(\d+)', id))
    model_filenames = [
        f for f in os.listdir(model_dir) if pattern.match(f)]
    if len(model_filenames) == 1:
        model_filenames = model_filenames[0]
    if not model_filenames:
        raise Exception(
            "Could not find any model summaries for the system"
            " summary with ID {}. Specified model filename pattern was: "
            "{}".format(id, model_filenames_pattern))
    return model_filenames

#taken from GitHub @bheinzerling
def convert_text_to_rouge_format(text, title="dummy title"):
    """
    Convert a text to a format ROUGE understands. The text is
    assumed to contain one sentence per line.
    text:   The text to convert, containg one sentence per line.
    title:  Optional title for the text. The title will appear
                in the converted file, but doesn't seem to have
                any other relevance.
    Returns: The converted text as string.
    """
    sentences = text.split("\n")
    sent_elems = [
        "<a name=\"{i}\">[{i}]</a> <a href=\"#{i}\" id={i}>"
        "{text}</a>".format(i=i, text=sent)
        for i, sent in enumerate(sentences, start=1)]
    html = """<html>
<head>
<title>{title}</title>
</head>
<body bgcolor="white">
{elems}
</body>
</html>""".format(title=title, elems="\n".join(sent_elems))

    return html

def write_text_with_rouge_format(system_dir, model_dir,original_name_ref = 'summary',original_name_cand = 'summary',converted_name_ref = 'ref.', converted_name_cand = 'cand.'):
    filenames_ref, documents_ref = read_directory (model_dir, name_text = original_name_ref)
    filenames_hyp, documents_hyp = read_directory (system_dir, name_text = original_name_cand)
    html_ref = []
    html_hyp = []
    id_ref = []
    id_hyp = []
    assert (len(filenames_ref)==len(filenames_hyp)),'ref summaries'+str(len(filenames_ref))+' hyp summaries'+str(len(filenames_hyp))

    for i in range (0,len(filenames_ref)):
        html_ref.append(convert_text_to_rouge_format(documents_ref[i], title="dummy title"))
        id_ref.append(extract_id(filenames_ref[i]))
        html_hyp.append(convert_text_to_rouge_format(documents_hyp[i], title="dummy title"))
        id_hyp.append(extract_id(filenames_hyp[i]))
    save_texts(model_dir, converted_name_ref, html_ref, id_ref)
    save_texts(system_dir, converted_name_cand, html_hyp, id_hyp)

def select_partition(directory,partition_file, partition = 'test', name_text = 'summary.'):
    '''parameters:
    directory : directory where documents are saved (with their index in the name)
    partition_file : json document with index of partition (training,validation, test or english, spanish, french, arabic, others)
    name_text: type of document to find (summary., document., ref.)
    '''
    if (partition == 'training' or partition == 'english'):
        part = 0
    elif (partition == 'validation' or partition == 'spanish'):
        part = 1
    elif (partition == 'test' or partition == 'french'):
        part = 2 
    elif (partition == 'arabic'):
        part = 3
    else:
        print('please provide a valid name of partition. If the partition is with respect to the training process: training or validation or test. If the partition is with respect to the language: english or  spanish or french or arabic.'), 
    
    if (type(partition_file) == list):
        data = partition_file
    else:
        with open(partition_file) as json_file:
            data = json.load(json_file)
    data = data[part]
    filename = []
    docum = []

    for i in range(0,len(data)):
        temp_fn, temp_d = read_directory(directory, name_text+data[i]+'.txt')
        #we are searching element by element(we don't want a list, we want the string name)
        filename.append(temp_fn[0])
        docum.append(temp_d[0])                                                                                                 
    return filename, docum


def identify_language(data_directory, name_text):
    fn, doc = read_directory(data_directory, name_text)
    en  = []
    es = []
    fr = []
    ar = []
    other = []
    no_lang = []
    #ids = extract_id(fn)
    ids = [sub.replace('summary.','') for sub in fn]
    ids = [sub.replace('.txt','') for sub in ids]
    for i in range (0,len(doc)):
        try:
            temp = detect(doc[i])
            if (temp == 'en'):
                en.append(ids[i])
            elif (temp == 'es'):
                es.append(ids[i])
            elif (temp == 'fr'):
                fr.append(ids[i])
            elif (temp == 'ar'):
                ar.append(ids[i])
            else:
                other.append(ids[i])
        except:
            no_lang.append(ids[i])
    result  = [en,es,fr,ar,other]
    return result, no_lang


def join_texts(directory, name_text, language):
    lang_part, _ = identify_language(directory, name_text)
    _, documents = select_partition(directory, lang_part, language, name_text)
    return ' '.join(documents)

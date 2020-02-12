#!/bin/bash

set -x
'''
for i; do
    echo $i
done
'''

nvidia-smi

sleep 5

export CUDA_VISIBLE_DEVICES=0

'''
DATAHOME=${@:(-2):1}
EXEHOME=${@:(-1):1}
'''

DATAHOME=/idiap/temp/jbello/data/training/neusum/
EXEHOME=/idiap/temp/jbello/models/neusum/NeuSum/neusum_pt/

ls -l ${DATAHOME}

ls -l ${EXEHOME}

SAVEPATH=/idiap/temp/jbello/training/neusum/neusum_random/sent3_100k_100d_200s/

mkdir -p ${SAVEPATH}

cd ${EXEHOME}

/idiap/temp/jbello/anaconda3/envs/nse/bin/python3 train.py -save_path ${SAVEPATH} \
                -online_process_data \
                -max_doc_len 80 \
                -train_oracle ${DATAHOME}/train.rouge_bigram_F1.oracle.regGain \
                -train_src ${DATAHOME}/train.src.txt \
                -train_src_rouge ${DATAHOME}/train.rouge_bigram_F1.oracle.regGain \
                -src_vocab ${DATAHOME}/neusum_random/sent3_100k_100d_200s/vocab.txt.100k \
                -train_tgt ${DATAHOME}/train.tgt.txt \
                -tgt_vocab ${DATAHOME}/neusum_random/sent3_100k_100d_200s/vocab.txt.100k \
                -layers 1 -word_vec_size 100 -sent_enc_size 256 -doc_enc_size 256 -dec_rnn_size 256 \
                -sent_brnn -doc_brnn \
                -dec_init simple \
                -att_vec_size 256 \
                -norm_lambda 20 \
                -sent_dropout 0.3 -doc_dropout 0.2 -dec_dropout 0\
                -batch_size 64 -beam_size 1 \
                -epochs 100 \
                -optim adam -learning_rate 0.001 -halve_lr_bad_count 100000 \
                -gpus 0 \
                -curriculum 0 -extra_shuffle \
                -start_eval_batch 1000 -eval_per_batch 1000 \
                -log_interval 100 -log_home ${SAVEPATH} \
                -seed 12345 -cuda_seed 12345 \
		-pre_word_vecs_enc /idiap/temp/jbello/data/training/neusum/neusum_random/sent3_100k_100d_200s/random.xavier.100d.txt \
		-freeze_word_vecs_enc \
                -dev_input_src /idiap/temp/jbello/data/validation/neusum/val.src.txt \
                -dev_ref /idiap/temp/jbello/data/validation/neusum/val.tgt.txt \
                -max_decode_step 3 -force_max_len

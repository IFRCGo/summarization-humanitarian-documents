
word_emb = '/idiap/temp/jbello/data/training/neusum_aspect/glove.100k.50d.txt'
asp_emb = '/idiap/temp/jbello/data/training/neusum_aspect/aspects.50d.txt'
save_path = '/idiap/temp/jbello/data/training/neusum_aspect/'

with open(word_emb,'r') as f:
    file = f.read()
    word =file.split('\n')

with open(asp_emb, 'r') as f:
    file = f.read()
    asp = file.split('\n')
print('Documents read!')
for aspect in asp:
    print('One aspect to save!')
    asp_name = aspect.split(' ')[0]
    asp_emb = aspect.replace(asp_name,'')
    word_asp = [line + asp_emb for line in word]
    short_path = save_path +'short_docs/'+asp_name+'/'+'glove.a.100k.100d.txt'
    long_path = save_path+'long_docs/'+asp_name+'/'+'glove.a.100k.100d.txt'
    temp_paths = [short_path, long_path]
    for temp_path in temp_paths:
        with(open(temp_path,'w', encoding = 'utf-8')) as f:
            for row in word_asp:
                f.write(str(row)+'\n')
print('everything saved!')


import re

with open("/idiap/temp/jbello/data/training/neusum/neusum_in_domain/sent4_10k_100d_80s/glove_emb.100d.txt","r") as txt_file:
    data = txt_file.read()
sp = re.sub('\[',' ', data)
sp = re.sub('\]',' ', sp)
sp = re.sub('\n -',' -', sp)
sp = re.sub('\n 0',' 0', sp)
sp = re.sub('\n 1',' 1', sp)
sp = re.sub('\n 2',' 2', sp)
sp = re.sub('\n 3',' 3', sp)
sp = re.sub('\n 4',' 4', sp)
sp = re.sub('\n 5',' 5', sp)
sp = re.sub('\n 6',' 6', sp)
sp = re.sub('\n 7',' 7', sp)
sp = re.sub('\n 8',' 8', sp)
sp = re.sub('\n 9',' 9', sp)
sp = re.sub('  ',' ', sp)
sp = re.sub('\n -',' -', sp)
sp = re.sub('\n 0',' 0', sp)
sp = re.sub('\n 1',' 1', sp)
sp = re.sub('\n 2',' 2', sp)
sp = re.sub('\n 3',' 3', sp)
sp = re.sub('\n 4',' 4', sp)
sp = re.sub('\n 5',' 5', sp)
sp = re.sub('\n 6',' 6', sp)
sp = re.sub('\n 7',' 7', sp)
sp = re.sub('\n 8',' 8', sp)
sp = re.sub('\n 9',' 9', sp)
#sp.count('\n')
sp = re.sub('  ',' ', sp)
sp = re.sub(' \n','\n', sp)
sp = re.sub('-0. ','0 ', sp)
sp = re.sub('  ',' ', sp)
sp = re.sub('  ',' ', sp)
sp = re.sub(' \n','\n', sp)
sp = re.sub('\n ','\n', sp)
sp = re.sub('  ',' ', sp)
sp = re.sub('\n\n','\n',sp)
#removing the ending indicator of new line
n = len(sp)
sp = sp[0:(n-2)]
sl = sp.split('\n')

count = 0
#to_del = []
for i in range(0,len(sl)):
    items = sl[i].split(' ')
    if (len(items) != 101):
        count = count + 1
        sp = sp.replace(sl[i],'')
        #to_del.append(i)

#to_del.reverse()
#for item in to_del:
#    del sl[item]

print('Embedding with errors: ', count)

#sp_mod = '\n'
#sp_mod.join(sl)
sp = sp.replace('\n\n','\n')
sp = sp.replace('\n \n','\n')
n = len(sp)
sp = sp[0:(n-2)]


sl = sp.split('\n')
count = 0
for line in sl:
    items = line.split(' ')
    print(len(items))
    count = count+1
    print(count)
    assert(len(items) == 101)
with open('/idiap/temp/jbello/data/training/neusum/neusum_in_domain/sent4_10k_100d_80s/glove.100d.txt','w') as txt_file:
    txt_file.write(sp)


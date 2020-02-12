import re

with open("/idiap/temp/jbello/data/training/neusum/neusum_random/sent3_100k_50d_80s/random.50d.txt","r") as txt_file:
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
#removing the ending indicator of new line
n = len(sp)
sp = sp[0:(n-2)]
sl = sp.split('\n')

for line in sl:
    items = line.split(' ')
    assert(len(items) == 51)
with open('/idiap/temp/jbello/data/training/neusum/neusum_random/sent3_100k_50d_80s/random.xavier.50d.txt','w') as txt_file:
    txt_file.write(sp)


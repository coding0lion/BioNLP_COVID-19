import re
import spacy
import pandas as pd
from collections import Counter

url_desk='C:/Users/ASUS/Desktop/'
file=open(url_desk+'result_0.txt')
cluster_0=file.read()
file.close()
file=open(url_desk+'result_1.txt')
cluster_1=file.read()
file.close()
file=open(url_desk+'result_2.txt')
cluster_2=file.read()
file.close()
cluster_0=cluster_0.split('\n')
cluster_0.append('cluster_0')
cluster_1=cluster_1.split('\n')
cluster_1.append('cluster_1')
cluster_2=cluster_2.split('\n')
cluster_2.append('cluster_2')

#确定了每篇文章的编号，下面进行分析
url_in='E:/大三下/自然语言处理/初步分析/data_new/'
entity_value={}

def interaction_get(data,array):    #后面的array是cluster_0/1/2
    if(data.split('|a|')[0] in array):
        data = data.split('\n')
        data_all = [i.split('|a|')[1] for i in data if '|a|' in i]
        data_all = [i + '.' for i in data_all]
        sentences = re.split(r' *[\.\?!][\'"\)\]]* *', data_all[0])
        for i in range(len(data)):
            if ('\t' in data[i]):
                data_entity = data[i].split('\t')
                entity = data_entity[3]
                if(data_entity[4] in ['Disease','Species','Chemical','Gene']):
                    for i in range(len(sentences)):
                        if (entity in sentences[i]):
                            if(entity not in list(entity_value.keys())):
                                entity_value[entity] = [sentences[i]+"\t"+data_entity[4]+"\t"+array[-1]]
                            if(entity in list(entity_value.keys())):
                                entity_value[entity].append(sentences[i]+"\t"+data_entity[4]+"\t"+array[-1])


for i in range(1,94848):
    file=open(url_in+'test_'+str(i))
    data=file.read()
    file.close()
    interaction_get(data,cluster_0)
    interaction_get(data,cluster_1)
    interaction_get(data,cluster_2)

#统计下总共多少句子
#num=0
#for i in range(len(list(entity_value.keys()))):
#    num=num+len(entity_value[list(entity_value.keys())[i]])
#得到结果是总句子数目

data_all=pd.DataFrame({'cluster':[0],'species':[0],'entity':[0]})
#已经提取，下面进行含新冠词语的依存关系查询，主要是对entity_value
nlp = spacy.load('en_core_web_sm')
n=0
for i in range(len(list(entity_value.keys()))):
    value=entity_value[list(entity_value.keys())[i]]
    entity=list(entity_value.keys())[i]
    for j in range(len(value)):
        word,entity_species,cluster=value[j].split('\t')
        doc=nlp(word)
        for token in doc:
            if((token.text == entity and token.head.text == 'COVID-19') or (token.text == 'COVID-19' and token.head.text == entity)):
                data_all.loc[n]=[cluster,entity_species,entity]
                n+=1


#end=list(entity_list.keys())
#end=[i+'\n' for i in end]
#file=open('C:/Users/ASUS/Desktop/result.txt','w')
#file.writelines(end)
#file.close()

data_all.to_csv('C:/Users/ASUS/Desktop/result.csv')

data_all=pd.read_csv('C:/Users/ASUS/Desktop/result.csv')
data_2=data_all[data_all.iloc[:,1]=='cluster_1']
data_cluster1=data_2[data_2.iloc[:,2]=='Species']

data_3=data_all[data_all.iloc[:,1]=='cluster_2']
data_cluster2=data_3[data_3.iloc[:,2]=='Disease']

data_cluster1_entity=list(data_cluster1.iloc[:,3])
data_cluster2_entity=list(data_cluster2.iloc[:,3])
Counter(data_cluster1_entity)
Counter(data_cluster2_entity)


#进行实体数目的再次统计
file=open('C:/Users/ASUS/Desktop/result_all.txt')
data=file.readlines()
file.close()
species_arr=[]
for i in range(len(data)):
    value=data[i].split('\t')
    species_arr.append(value[4])
Counter(species_arr)
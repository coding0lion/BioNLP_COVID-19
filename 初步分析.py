#这里换个切分文件的方式
#①采用liunx指令csplit（需要安装）②以|t|作为切分标准
url_in='C:/Users/ASUS/Desktop/初步分析/data/'
url_out='C:/Users/ASUS/Desktop/初步分析/data_new/'
name_poc=[]
for i in range(97,102):
    for j in range(97,123):
        for k in range(97,123):
            var='test_a'+chr(i)+chr(j)+chr(k)
            name_poc.append(var)
i=102        #f
for j in range(97,103): #f
    for k in range(97,123):
        var = 'test_a' + chr(i) + chr(j) + chr(k)
        name_poc.append(var)
j=103   #g
for k in range(97,105):    #h
    var = 'test_a' + chr(i) + chr(j) + chr(k)
    name_poc.append(var)

cache_value=[]    #这是一个数据缓存变量，用以存储下一个文件的内容
title_all=[]
#记录下待索引的文件名称,已确认
id_value=0   #用于组成文章的名称
for i in range(len(name_poc)):
    url=url_in+name_poc[i]
    file=open(url)
    data=file.read()
    file.close()
    data=data.split('\n')
    for j in range(len(data)):
        if('|t|' in data[j]):
            url_output=url_out+'test_'+str(id_value)
            f=open(url_output,'w')
            f.writelines(cache_value)
            f.close()
            cache_value=[]
            title_all.append(data[j])
            id_value+=1
        elif('|t|' not in data[j]):
            cache_value.append(data[j]+'\n')

#文章标题
title=[]
for i in range(len(title_all)):
    data_0,data_1=title_all[i].split('|t|')
    title.append(data_0+'\t'+data_1+'\n')
file=open('C:/Users/ASUS/Desktop/标题信息.txt','w')
file.writelines(title)
file.close()

#已确认提取

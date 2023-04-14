# 导入相关库
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.cluster import KMeans
import pandas as pd
from sklearn.preprocessing import normalize
from transformers import AutoTokenizer, BioGptForCausalLM

def pre_data(df,n):
    id=df.iloc[:,1]
    id = id.tolist()
    idd = list(set(id))
    idd=sorted(idd)
    #计算每篇文章实体的的it-idf值
    a = []
    for i in idd:
        idex = []
        idex = index_all(i,id)
        ents=df.iloc[idex,n]
        strd = ents.str.cat(sep=' ')#转换为一行字符串
        a.append(strd)
    return a

def index_all(numd,lst):
    indices = []
    start = 0
    end = len(lst)
    try:
        while True:
            idx = lst.index(numd,start,end)
            indices.append(idx)
            start = idx + 1
    except ValueError:
        pass
    return indices

def countvect(a):
    from sklearn.feature_extraction.text import CountVectorizer
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(a)
    print(vectorizer.get_feature_names())
    print(X.toarray())
    count = X.toarray()
    return count

raw = pd.read_csv('result_all.txt', delimiter='\t',header=None)
title_raw = pd.read_csv('标题信息.txt', delimiter='\t',header=None)
df = raw.copy()
# 读取文献标题数据，假设每行一个标题，存储在titles.txt文件中
#title_raw.drop_duplicates(subset=title_raw.columns[0], keep='first', inplace=True)#去除标题数据，id重复
#merged_df = pd.merge(raw, title_raw, on=raw.columns[0] and title_raw.columns[0], how='inner')#匹配上
#merged_df.drop_duplicates(subset=merged_df.columns[0], keep='first', inplace=True)


merged_df = pd.read_csv('all_titles.csv')
title=merged_df.iloc[:,6]
titles = title.tolist()


#enti_type=pre_data(df,5)
#count=countvect(enti_type)
#count = normalize (count, axis=0) # 对列进行标准化

import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch

# Load pre-trained model tokenizer (vocabulary)
tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-v1.1")

# Load pre-trained model (weights)
model = AutoModel.from_pretrained("dmis-lab/biobert-v1.1")

batch_size = 1024
max_length = 256
title_embeddings = []
for i in range(0, len(titles), batch_size):
    batch_titles = titles[i:i+batch_size]
    title_ids = tokenizer(batch_titles, padding=True, truncation=True, max_length=max_length, return_tensors="pt")["input_ids"]
    with torch.no_grad():
        batch_embeddings = model(title_ids)[1].numpy()
    title_embeddings.append(batch_embeddings)
    print("Finished:",i/len(titles))
title_embeddings = np.concatenate(title_embeddings, axis=0)
print(title_embeddings.shape)

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
#X= np.concatenate ( (title_embeddings, count), axis=1) # 沿着第0轴（行）合并

X = title_embeddings

for n_clusters in [2,3,4,5,6,7]:
    n_clusters = n_clusters
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)
    ax1.set_xlim([-0.1, 1])
    ax1.set_ylim([0, X.shape[0] + (n_clusters + 1) * 10])
    clusterer = KMeans(n_clusters=n_clusters, random_state=10).fit(X)
    cluster_labels = clusterer.labels_
    silhouette_avg = silhouette_score(X, cluster_labels)
    print("For n_clusters =", n_clusters,
          "The average silhouette_score is :", silhouette_avg)
    sample_silhouette_values = silhouette_samples(X, cluster_labels)
    y_lower = 10
    for i in range(n_clusters):
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
        ith_cluster_silhouette_values.sort()
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
        color = cm.nipy_spectral(float(i)/n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper)
                          ,ith_cluster_silhouette_values
                          ,facecolor=color
                          ,alpha=0.7
                         )
        ax1.text(-0.05
                 , y_lower + 0.5 * size_cluster_i
                 , str(i))
        y_lower = y_upper + 10

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
    ax1.set_yticks([])
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
    ax2.scatter(X[:, 0], X[:, 1]
                ,marker='o'
                ,s=8
                ,c=colors
               )
    centers = clusterer.cluster_centers_
    # Draw white circles at cluster centers
    ax2.scatter(centers[:, 0], centers[:, 1], marker='x',
                c="red", alpha=1, s=200)

    ax2.set_title("The visualization of the clustered data.")
    ax2.set_xlabel("Feature space for the 1st feature")
    ax2.set_ylabel("Feature space for the 2nd feature")

    plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                  "with n_clusters = %d" % n_clusters),
                 fontsize=14, fontweight='bold')
    plt.savefig ('EMG {0}.jpg'.format (i))
    plt.show()

# 导入需要的库
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

# 使用kmeans对高维数据进行聚类，指定簇的数量为4
kmeans = KMeans(n_clusters=2, random_state=4)
kmeans.fit(title_embeddings)
labels = kmeans.labels_
label = pd.DataFrame(labels)

# 使用to_csv()方法保存文件为csv
label.to_csv('label.csv')
# 使用t-SNE对高维数据进行降维，指定目标维度为2
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X)

# 使用matplotlib对降维后的数据进行可视化，根据聚类标签或簇标签给不同的数据点分配不同的颜色和形状
plt.figure(figsize=(8, 6))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels, marker='o', cmap='rainbow')
#plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, marker='x', cmap='rainbow')
plt.xlabel('t-SNE dimension 1')
plt.ylabel('t-SNE dimension 2')
plt.title('t-SNE visualization of kmeans clustering results')
plt.savefig ('visable.jpg')
plt.show()


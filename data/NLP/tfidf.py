from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

with open('subjectsNamesInFinalGrades.txt', 'r') as f:
    corpus = f.readlines()

k=0
for i in corpus:
    corpus[k] = i.strip('\n')
    k = k+1

vectorizer = TfidfVectorizer()

# TD-IDF Matrix
X = vectorizer.fit_transform(corpus)

# extracting feature names
tfidf_tokens = vectorizer.get_feature_names_out()

# reduce the dimensionality of the data using PCA
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(X.toarray())

# cluster the documents using k-means
num_clusters = 4
kmeans = KMeans(n_clusters=num_clusters, init = 'k-means++', n_init=5,  
                max_iter=500, random_state=42)
kmeans.fit(X)

# create a dataframe to store the results   
results = pd.DataFrame()
results['document'] = corpus
results['cluster'] = kmeans.labels_

# print the results
print(results.sample(5))


# plot the results
colors = ['red', 'green', 'blue', 'yellow']
cluster = ['1','2','3','4']
for i in range(num_clusters):
    plt.scatter(reduced_data[kmeans.labels_ == i, 0],
                reduced_data[kmeans.labels_ == i, 1],
                s=10, color=colors[i],
                label=f' {cluster[i]}')
plt.legend()
plt.show()

labels = kmeans.labels_ 
corpus['label'] = labels
corpus.head()




def get_top_features_cluster(tf_idf_array, prediction, n_feats):
    labels = np.unique(prediction)
    dfs = []
    for label in labels:
        id_temp = np.where(prediction==label) # indices for each cluster
        x_means = np.mean(tf_idf_array[id_temp], axis = 0) # returns average score across cluster
        sorted_means = np.argsort(x_means)[::-1][:n_feats] # indices with top 20 scores
        features = vectorizer.get_feature_names()
        best_features = [(features[i], x_means[i]) for i in sorted_means]
        df = pd.DataFrame(best_features, columns = ['features', 'score'])
        dfs.append(df)
    return dfs

def plotWords(dfs, n_feats):
    plt.figure(figsize=(8, 4))
    for i in range(0, len(dfs)):
        plt.title(("Most Common Words in Cluster {}".format(i)), fontsize=10, fontweight='bold')
        sns.barplot(x = 'score' , y = 'features', orient = 'h' , data = dfs[i][:n_feats])
        plt.show()
best_result = 5
kmeans = kmeans_results.get(best_result)

final_df_array = final_df.to_numpy()
prediction = kmeans.predict(final_df)
n_feats = 20
dfs = get_top_features_cluster(final_df_array, prediction, n_feats)
plotWords(dfs, 13)




import pandas as pd

result = pd.DataFrame(
    data=X.toarray(), 
    index=[f"Doc{i}" for i in range(len(corpus))], 
    columns=tfidf_tokens
)

result.to_csv('TF-IDF.csv')
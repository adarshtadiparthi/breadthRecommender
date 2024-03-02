import pandas as pd

cluster_names = pd.read_csv('ClusterNames.csv')

cluster_names.drop('Unnamed: 0',axis=1,inplace=True)
cluster_names.columns = ['Profile Names']

with open('subjectsNamesInFinalGrades.txt', 'r') as f:
    subject_names = f.readlines()

k=0
for i in subject_names:
    subject_names[k] = i.strip('\n')
    k = k+1
    
subject_names = pd.DataFrame(subject_names)
subject_names.columns = ['Subject Names']

subject_names_with_clusters = pd.concat([subject_names, cluster_names],axis=1)

subject_names_with_clusters.rename(columns = {'0':'Profile Names'}, inplace = True) 

subject_names_with_clusters.to_csv('Subject Names With Clusters.csv')
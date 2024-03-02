import pandas as pd
import numpy as np

subject_names_with_clusters = pd.read_csv('Subject Names With Clusters.csv')
final_grades = pd.read_csv('divided final_grades.csv')

subject_names_with_clusters.drop('Unnamed: 0',axis=1,inplace=True)

subject_names_with_clusters.set_index('Subject Names')

list_of_subjects = subject_names_with_clusters['Subject Names'].to_list()

final_grades.drop('Unnamed: 0',axis=1,inplace=True)

final_grades.insert(2,"Profile", None, True)

# subject_names_with_clusters[final_grades['Course Names'].isin(list_of_subjects)]['Profile Names']

# final_grades_new = pd.DataFrame()
# final_grades.sort_values(by=['Course Names','session'],ascending=False,inplace=True)
# unique_names = final_grades['Course Names'].unique()

result = final_grades.sort_values(by=['Course Names','session']).drop_duplicates(subset='Course Names', keep='first')

result.set_index('Course Names',inplace=True)

for i in list_of_subjects:
    result.loc[i,'Profile'] = subject_names_with_clusters['Profile Names'][list_of_subjects.index(i)]

result = result.sort_values(by='Avg CGPA')

result.replace('#DIV/0!', np.nan, inplace=True)

result.dropna(inplace=True)

result.to_csv('final_grades along with profile.csv')

def f(x):
    return x

f(result)
import pandas as pd

data = pd.read_csv('oldGrades.csv')

data.set_index('subject', inplace=True)

with open('subjectNos.txt', 'r') as f:
    subject_names = f.readlines()

k=0
for i in subject_names:
    subject_names[k] = i.strip('\n')
    k = k+1

data_new = pd.DataFrame()
for i in range(len(subject_names)):
    try:
        data_new = data_new._append(data.loc[subject_names[i]])
        print([subject_names[i]])
    except:
        continue
    
data_new.to_csv('Final_grades&BreadthList.csv', index=True)
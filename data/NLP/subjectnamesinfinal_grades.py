import pandas as pd

data = pd.read_csv('../final_grades.csv')

course = data.pop('course')

course_list = course.values.tolist()

course_list_new =[]
course_index_list = []
course_code_list = []
for i in course_list:
    try:
        if i!='subject subjec':
            course_code_list.append(i.split('-')[0])
        else:
            data.drop(course_list.index(i),axis=0,inplace=True)
            data.reset_index(drop=True,inplace=True)
        if i.split('-')[1]!='':
            course_list_new.append(i.split('-')[1:])
        else:
            course_list_new.append('')
            continue
    except:
        continue
    
course_code = pd.DataFrame(course_code_list)
course_code.columns=['Course Code']

course_list_new_1 = []
for i in course_list_new:
    j = ' '.join(i)
    course_list_new_1.append(j)
course_new = pd.DataFrame(course_list_new_1)
course_new.columns = ['Course Names']
course_list_new_1 = list(set(course_list_new_1))
    
with open('subjectsNamesInFinalGrades.txt', 'w') as f:
    for line in course_list_new_1:
        f.write(f"{line}\n")
        
course = pd.concat([course_code,course_new],axis=1)

data = pd.concat([course,data],axis=1)

data.to_csv('divided final_grades.csv')
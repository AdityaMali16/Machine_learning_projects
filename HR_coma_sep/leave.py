import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model


df= pd.read_csv('HR_coma_sep/HR_comma_sep.csv')
print(df.columns)

left=df[df['left']==1]
print(left.shape)
working=df[df['left']==0]
print(working.shape)

print(df.groupby('left').mean(numeric_only=True))

ct_salary=pd.crosstab(df['salary'],df['left']).plot(kind='bar')
ct_department=pd.crosstab(df['Department'],df['left']).plot(kind='bar')
plt.show()



X=df.drop(['last_evaluation', 'number_project', 'time_spend_company'],axis='columns')
y=df['left']
dummies=pd.get_dummies(df[['salary','Department']],drop_first=True)
X = pd.concat([
    df[['satisfaction_level',
        'average_montly_hours',
        'time_spend_company',
        'Work_accident',
        'promotion_last_5years']],
    dummies
], axis='columns')



from sklearn.model_selection import train_test_split
X_train , X_test , Y_train , Y_test=train_test_split(X,y,test_size=0.3)

model=linear_model.LogisticRegression(max_iter=5000)
model.fit(X_train,Y_train)

print(model.predict(X_test))


print(model.score(X,y))

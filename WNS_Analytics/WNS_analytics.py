import pandas as pd
from xgboost import XGBClassifier

def clean_data(df):
    del_cols=['gender','employee_id','age','length_of_service','no_of_trainings']
    df.drop(del_cols,axis=1,inplace=True)
    col_list = ['department','region','education','recruitment_channel']
    for i in col_list:
        df=pd.concat([df,pd.get_dummies(df[i])],axis=1)
        df.drop(i,axis=1, inplace=True)
    df = df.fillna(df.mean())
    return df.values

    
col_names=[]
train=pd.read_csv('train.csv')
test=pd.read_csv('test.csv')
y = train['is_promoted'].values
train.drop('is_promoted',axis=1,inplace=True)
eid=test['employee_id'].tolist()
X=clean_data(train)
X_test=clean_data(test)
seed = 7
model = XGBClassifier(learning_rate=0.9,n_estimators=300)
model.fit(X, y)
y_pred=model.predict(X_test)
output = pd.DataFrame({'employee_id':eid,'is_promoted':y_pred})
output.to_csv('submit.csv',index=False)


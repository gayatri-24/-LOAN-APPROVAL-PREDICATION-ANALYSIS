#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
A = pd.read_csv("C:/Users/Gayatri/Downloads/training_set.csv")


# In[2]:


A.head()


# # Data Preprocessing
# Treating credit history as catgorical

# In[3]:


A['Credit_History'].value_counts()


# In[4]:


A['Credit_History']=A['Credit_History'].fillna(1.0)


# In[5]:


A.isna().sum()


# In[6]:


def replacer(df):
    Q = pd.DataFrame(df.isna().sum())
    Q.columns=["CT"]
    w = list(Q[Q.CT>0].index)
    
    cat = []
    con = []
    for i in w:
        if(df[i].dtypes=="object"):
            cat.append(i)
        else:
            con.append(i)

    for i in con:
        replacer = df[i].mean()
        df[i] = df[i].fillna(replacer)

    for i in cat:
        replacer = pd.DataFrame(df[i].value_counts()).index[0]
        df[i] = df[i].fillna(replacer)


# In[7]:


A.info()


# In[8]:


A.isna().sum()


# In[9]:


replacer(A)


# In[10]:


A.isna().sum()


# In[11]:


A


# In[12]:


A.info()


# In[13]:


dep = []
for i in A.Dependents:
    dep.append(int(i.replace("3+","3")))
A.Dependents=dep


# In[14]:


A.info()


# In[15]:


cat = []
con = []
for i in A.columns:
    if(A[i].dtypes=="object"):
        cat.append(i)
    else:
        con.append(i)


# In[16]:


cat


# In[17]:


con


# In[18]:


cat.remove("Loan_ID")
cat.remove("Loan_Status")


# In[19]:


X = A[con].join(pd.get_dummies(A[cat]))


# In[20]:


X


# # EDA

# In[21]:


con


# In[23]:


from statsmodels.api import OLS
from statsmodels.formula.api import ols
model = ols("Credit_History ~ Loan_Status",A).fit()
from statsmodels.stats.anova import anova_lm
anova_results = anova_lm(model)
Q = pd.DataFrame(anova_results)
a = Q['PR(>F)']['Loan_Status']
print("%.40f"%a)


# In[ ]:


Logistic|NB. ----- DT|RF|AB


# In[24]:


Y = A[["Loan_Status"]]
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(X,Y,test_size=0.2,random_state=33)


# In[25]:


Y['Loan_Status'].value_counts().plot(kind="bar")


# In[26]:


ytrain['Loan_Status'].value_counts().plot(kind="bar")


# In[27]:


from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
model = lr.fit(xtrain,ytrain)
pred = model.predict(xtest)
from sklearn.metrics import accuracy_score
print("%.4f"%accuracy_score(ytest,pred))


# In[28]:


X.columns


# In[29]:


#X = X.drop(labels=["Dependents"],axis=1)
Y = A[["Loan_Status"]]
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(X,Y,test_size=0.2,random_state=33)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
model = lr.fit(xtrain,ytrain)
pred = model.predict(xtest)
from sklearn.metrics import accuracy_score
print("%.4f"%accuracy_score(ytest,pred))


# In[30]:


X = X.drop(labels=["ApplicantIncome"],axis=1)
Y = A[["Loan_Status"]]
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(X,Y,test_size=0.2,random_state=33)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
model = lr.fit(xtrain,ytrain)
pred = model.predict(xtest)
from sklearn.metrics import accuracy_score
print("%.4f"%accuracy_score(ytest,pred))


# In[31]:


Y = A[["Loan_Status"]]
X = A[con].join(pd.get_dummies(A[cat]))
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(X,Y,test_size=0.2,random_state=33)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
model = lr.fit(xtrain,ytrain)
pred = model.predict(xtest)
from sklearn.metrics import accuracy_score
print("%.4f"%accuracy_score(ytest,pred))


# In[32]:


Y = A[["Loan_Status"]]
X = A[con].join(pd.get_dummies(A[cat]))
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(X,Y,test_size=0.2,random_state=33)
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(random_state=11,max_depth=2)
model = dtc.fit(xtrain,ytrain)
pred = model.predict(xtest)
from sklearn.metrics import accuracy_score
print("%.4f"%accuracy_score(ytest,pred))


# In[33]:


Y = A[["Loan_Status"]]
X = A[con].join(pd.get_dummies(A[cat]))
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(X,Y,test_size=0.2,random_state=33)
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(random_state=11,max_leaf_nodes=2)
model = dtc.fit(xtrain,ytrain)
pred = model.predict(xtest)
from sklearn.metrics import accuracy_score
print("%.4f"%accuracy_score(ytest,pred))


# In[35]:


Y = A[["Loan_Status"]]
X = A[con].join(pd.get_dummies(A[cat]))
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(X,Y,test_size=0.2,random_state=33)
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(random_state=11,max_leaf_nodes=7,n_estimators=10)
model = rfc.fit(xtrain,ytrain)
pred = model.predict(xtest)
from sklearn.metrics import accuracy_score
print("%.4f"%accuracy_score(ytest,pred))


# In[36]:


Y = A[["Loan_Status"]]
X = A[con].join(pd.get_dummies(A[cat]))
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(X,Y,test_size=0.2,random_state=33)
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
dtc1 = DecisionTreeClassifier(max_depth=2,random_state=14)
abc = AdaBoostClassifier(dtc1,random_state=228,n_estimators=2)
model = abc.fit(xtrain,ytrain)
pred = model.predict(xtest)
from sklearn.metrics import accuracy_score
print("%.4f"%accuracy_score(ytest,pred))


# In[37]:


Y = A[["Loan_Status"]]
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
C = pd.DataFrame(ss.fit_transform(A[con]),columns=con)
X = C.join(pd.get_dummies(A[cat]))
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(X,Y,test_size=0.2,random_state=33)
from sklearn.neighbors import KNeighborsClassifier
knc = KNeighborsClassifier(n_neighbors=5)
model = knc.fit(xtrain,ytrain)
pred = model.predict(xtest)
from sklearn.metrics import accuracy_score
print("%.4f"%accuracy_score(ytest,pred))


# # Train with entire data on selected algo

# In[38]:


Y = A[["Loan_Status"]]
X = A[con].join(pd.get_dummies(A[cat]))
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(X,Y,test_size=0.2,random_state=33)
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
dtc1 = DecisionTreeClassifier(max_depth=2,random_state=14)
abc = AdaBoostClassifier(dtc1,random_state=228,n_estimators=2)
model = abc.fit(X,Y)


# In[39]:


Q= pd.read_csv("C:/Users/Gayatri/Downloads/testing_set.csv")


# In[40]:


Q.head(2)


# In[41]:


X.head(2)


# In[42]:


Q['Credit_History']=Q['Credit_History'].fillna(1.0)
replacer(Q)

dep = []
for i in Q.Dependents:
    dep.append(int(i.replace("3+","3")))
Q.Dependents=dep

R = Q[con].join(pd.get_dummies(Q[cat]))


# In[43]:


pred = model.predict(R)


# In[44]:


OP = Q[['Loan_ID']]


# In[45]:


OP['predicted_loan_status']=pred


# # Not eligible for loan

# In[46]:


NE = OP[OP.predicted_loan_status=="N"]


# In[47]:


R['pred_LS']=pred


# In[48]:


S = R[R.pred_LS=="N"]


# In[49]:


S.head(2)


# In[50]:


#S = S.drop(labels=["LoanAmount"],axis=1)
S = S.drop(labels=["pred_LS"],axis=1)


# In[51]:


M= A[A['Loan_Status']=="Y"]
Y = M[["LoanAmount"]]
X = M[con].join(pd.get_dummies(M[cat]))
X = X.drop(labels=["LoanAmount"],axis=1)
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
dtc1 = DecisionTreeRegressor(max_depth=2,random_state=14)
abc = AdaBoostRegressor(dtc1,random_state=228,n_estimators=2)
model = abc.fit(X,Y)


# In[53]:


pred_LA = model.predict(S)


# In[55]:


Y.shape


# In[56]:


Y.index = range(0,422)


# In[57]:


len(pred_LA)


# In[58]:


X.shape


# In[59]:


Y.shape


# In[60]:


S.index = range(0,63)


# In[61]:


pred_LA = model.predict(S)


# In[62]:


S['Pred_LA']=pred_LA


# In[63]:


S1 = R[R.pred_LS=="N"]
S1.index = range(0,63)


# In[64]:


S.head(2)


# # People who loans were rejected and Loan term was <= 20 yrs

# In[65]:


prediction_data = R[R.pred_LS=="N"][R.Loan_Amount_Term<=240]
prediction_data.index=range(0,5)


# In[66]:


prediction_data


# In[ ]:


training data


# In[68]:


import pandas as pd
A = pd.read_csv("C:/Users/Gayatri/Downloads/training_set.csv")
A['Credit_History']=A['Credit_History'].fillna(1.0)
replacer(A)


# In[69]:


A = A[A.Loan_Status=="Y"]


# In[70]:


A.head(3)


# In[71]:


A = A.drop(labels=['Loan_ID','Loan_Status'],axis=1)


# In[72]:


A.index = range(0,422)


# In[73]:


cat = []
con = []
for i in A.columns:
    if(A[i].dtypes=="object"):
        cat.append(i)
    else:
        con.append(i)
A = A[con].join(pd.get_dummies(A[cat]))
#training_data
Y = A[["Loan_Amount_Term"]]
X = A.drop(labels=['Loan_Amount_Term'],axis=1)


# In[74]:


Y


# In[75]:


from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
dtc1 = DecisionTreeRegressor(max_depth=2,random_state=14)
abc = AdaBoostRegressor(dtc1,random_state=228,n_estimators=2)
X = X.drop(labels=['Dependents_0', 'Dependents_1', 'Dependents_2', 'Dependents_3+'],axis=1)
model = abc.fit(X,Y)


# In[76]:


#prediction_data=prediction_data.drop(labels=["pred_LS"],axis=1)
#prediction_data=prediction_data.drop(labels=["Dependents",'Loan_Amount_Term'],axis=1)
model.predict(prediction_data)


# In[77]:


prediction_data


# In[ ]:





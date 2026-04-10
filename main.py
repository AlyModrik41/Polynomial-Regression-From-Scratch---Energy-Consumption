import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import seaborn as sns

def Polynomial_Transform(X,degree):
    X_arr=np.array(X)
    n_features=X_arr.shape[1]

    combos=get_combinations(n_features,degree)
    new_cols=[]

    for i in range(n_features):
        new_cols.append(X_arr[:,i])

    for degree in range(2,degree+1):
        combos=get_combinations(n_features,degree)
        for combo in combos:
            col = np.ones(X_arr.shape[0])
            for index in combo:
                col=col*X_arr[:,index]
            new_cols.append(col)

    return np.column_stack(new_cols)


def get_combinations(n_features,degree):
    combinations=[]
    current=[]
    for i in range(n_features):
        current.append((i,))
    for i in range(degree-1):
        next_level=[]
        for combo in current:
            last_number=combo[-1]
            for feature_index in range(last_number,n_features):
                new_combo=combo+(feature_index,)
                next_level.append(new_combo)
        current=next_level
    return current


def compute_mse(y_true,y_pred):
    y_true=np.array(y_true)
    y_pred=np.array(y_pred)

    mse=[]
    for i in range(0,len(y_true)):
        error=y_true[i]-y_pred[i]
        error=error*error
        mse.append(error)
    error=np.mean(np.array(mse))

    return error

def run_experiment(X_train,X_test,y_train,y_test,max_degree):
    mse_list=[]
    regularized_mse_list=[]
    mse_train_list=[]
    mse_lasso_list=[]
    for degree in range(1,max_degree+1):
        X_train_poly=Polynomial_Transform(X_train,degree)
        X_test_poly=Polynomial_Transform(X_test,degree)

        model=linear_model.LinearRegression()
        model.fit(X_train_poly,y_train)


        y_pred=model.predict(X_test_poly)

        mse=compute_mse(y_test,y_pred)
        mse_list.append(mse)

        mse_train=compute_mse(y_train,model.predict(X_train_poly))
        mse_train_list.append(mse_train)

        regularized_model=linear_model.Ridge(alpha=150)
        regularized_model.fit(X_train_poly,y_train)
        
        y_pred_reg=regularized_model.predict(X_test_poly)
        mse_reg=compute_mse(y_test,y_pred_reg)
        regularized_mse_list.append(mse_reg)
        
        Lasso_model=linear_model.Lasso(alpha=150)
        Lasso_model.fit(X_train_poly,y_train)

        y_pred_lasso=Lasso_model.predict(X_test_poly)

        mse_lasso=compute_mse(y_test,y_pred_lasso)
        mse_lasso_list.append(mse_lasso)
        
        print(f"Degree {degree} -> Train MSE : {mse_train} | Test MSE : {mse} | Ridge MSE : {mse_reg} | Lasso MSE : {mse_lasso} ")

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(range(1, max_degree + 1), mse_list, label='Test MSE',marker='s')
    ax.plot(range(1, max_degree + 1), regularized_mse_list, label='Ridge',marker='o')
    ax.plot(range(1,max_degree+1),mse_train_list,label='Train MSE',marker='d')
    ax.plot(range(1,max_degree+1),mse_lasso_list,label='Lasso',marker='*')
    ax.set_xlabel('Degree')
    ax.set_ylabel('MSE')
    ax.set_title('Polynomial Degree vs MSE')
    ax.legend()

    plt.tight_layout()
    plt.show()
    
    return mse_list


df=pd.read_csv('assignment1dataset.csv')
# print(df.isnull().sum())
# print(df.shape)
# df.dropna(inplace=True)
# print(df.isnull().sum())
# print(df.shape)
# print(df.columns)
# print(df['Position'].value_counts())
# df['is_gk']=(df['Position']=='GK').astype(int)
# df.drop(columns=['Club','Name','Nationality','Position'],inplace=True)
# print(df.head(-3))
# plt.figure(figsize=(20,15))
# sns.heatmap(df.corr(),annot=True,cmap='BuPu')
# plt.show()
# df.drop(columns=['GKDiving','GKHandling','GKKicking','Age',
#                  'SlidingTackle','StandingTackle','Interceptions','ShortPassing',
#                  'Dribbling','Crossing','FKAccuracy','LongShots','Acceleration',
#                  'Balance','ShotPower','Finishing','Jumping','Positioning',
#                 'Curve','Penalties','Reactions','SprintSpeed','Strength',
#                 'BallControl','Marking',],inplace=True)
# df=df[['Overall','Potential','Composure','Vision','LongPassing','Volleys','Agility','Value']]

# feature_cols=['Overall','Potential','Composure','Vision','LongPassing','Volleys','Agility']
df=df.drop(columns=['Average Temperature'])

feature_cols=['Square Footage','Number of Occupants','Appliances Used']

plt.figure(figsize=(10,5))
sns.heatmap(df.corr(),annot=True,cmap='BuPu')
plt.show()
# print(df.shape)

X=df[feature_cols]
y=df['Energy Consumption']

scaler=StandardScaler()


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)
run_experiment(X_train_scaled,X_test_scaled,y_train,y_test,9)









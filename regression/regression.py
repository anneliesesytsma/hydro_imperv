import numpy as np  
import matplotlib.pyplot as plt 
from pandas import *
import seaborn as sns
import pandas as pd  
import pickle
import glob
import os, sys
import string

from sklearn.pipeline import Pipeline
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import feature_selection
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor  
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_predict


def combine(soil):
    f=['SAT','FC','WP']
    core=[]

    for index,f in enumerate(f):
        run = 'SCSTYPEI_MGA_'+str(f)
        path =r'out/'+str(run)+''
        print(path)
        allFiles = glob.glob(path + "/*.csv")
        frame = pd.DataFrame()
        list_ = []

        for file_ in allFiles:
            df = pd.read_csv(file_,index_col=0)
            df['Source'] = os.path.basename(file_)
            list_.append(df)
        core_temp = pd.concat(list_)
        core_temp['IF']=core_temp['IF'].astype(float).round(3)
        core_temp['f_var']=1-core_temp['IF']
        core_temp['L']=core_temp['A']*43560/core_temp['W']
        core_temp['L:W']=core_temp['L']/core_temp['W']
        core_temp['Ks_mm']=(core_temp['Ks']*25.4).astype(float).round(1)
        core_temp['P_cm']=(core_temp['P']*2.54).astype(float).round(1)
        core_temp['H_i_cm']=(core_temp['H_i']*2.54).astype(float).round(1)
        core_temp['SoilConditions']=str(f)
        core_temp['IMD']=(core_temp['IMD']).astype(float).round(1)
        core_temp['A_hec']=core_temp['A']*0.404686
        core_temp['W_m']=core_temp['W']*0.3048
        core_temp['L_m']=(core_temp['A_hec']/0.0001)/core_temp['W_m']
        core_temp['A_check']=core_temp['W_m']*core_temp['L_m']*0.0001
        core_temp['W_norm']=np.log(core_temp['W_m'])
        soil['Ks']=soil['K']
        soil_ks=soil[['Ks','Soil Texture Class']]
        core_temp=core_temp.merge(soil_ks, how='left',on='Ks')
        core_temp['soil']=core_temp['Soil Texture Class']
        core_temp=core_temp[['W_m','L:W','W','W_norm','A_hec','fV','S','P_cm','f_var','Ks_mm','SoilConditions','soil','H_i_cm']]

        core.append(core_temp)

    core = pd.concat(core)
    with open('out/current_run/CORE_ALL.pickle', 'wb') as handle:
        pickle.dump(core, handle, protocol=pickle.HIGHEST_PROTOCOL)
    core = pd.DataFrame(core)
    
    return core

def bins(core):
    
    bins=np.linspace(0, 1., 11).astype(float).tolist()
    W_bins=np.linspace((core['W_norm'].min()).astype(float).round(2), 
                       (core['W_norm'].max()).astype(float).round(2), 11).astype(float).tolist()

    labels = ','.join(str(e) for e in bins)[4:].split(",")
    W_labels = ','.join(str(e) for e in W_bins)[6:].split(",")
    core['fV_bins'] = pd.cut(x=core['fV'], bins=bins, labels=labels).astype(float).round(2) 
    core['a_bins'] = pd.cut(x=core['A_hec'], bins=bins, labels=labels).astype(float).round(2) 
    core['w_bins'] = pd.cut(x=core['W_norm'], bins=W_bins, labels=W_labels).astype(float).round(2) 

    return core

def boxplots(x,y):
    
    n=x*y
    'n is number of subplots'
    
    f, axes = plt.subplots(x,y,sharey=True,figsize= (5*x,4*y))
    sns.set(style='whitegrid', rc={"grid.linewidth": 0.1})
    f.subplots_adjust(left=0.09, bottom=0.1, right=0.99, top=0.99, wspace=0.2)
    f.subplots_adjust(hspace = 0.4)
    sns.set_context("paper", font_scale=1.7) 
    
    labels=list(string.ascii_uppercase)
    n_list=list(range(n))
    ab = zip(labels, n_list)
    
    for i,(a,b) in enumerate(ab):
        f.text(0.0, 1.01*(n-0.95*i)/n, a, weight="bold", horizontalalignment='left', verticalalignment='center')
    return f, axes


def processing_cat(core_df):

    X = core_df.select_dtypes(include=[object])
    # encode labels with value between 0 and n_classes-1.
    le = preprocessing.LabelEncoder()
    # use df.apply() to apply le.fit_transform to all columns
    X_2 = X.apply(le.fit_transform)
    enc = preprocessing.OneHotEncoder(categories='auto').fit(X_2)
    onehotlabels = enc.transform(X_2).toarray()
    soilc_dummies = pd.get_dummies(core_df.SoilConditions)
#     soil_dummies = pd.get_dummies(core_df.soil)

    df=pd.concat([core_df, soilc_dummies], axis=1)
#     df_final=pd.concat([df, soil_dummies], axis=1)
    df = df.select_dtypes(exclude=['category','object'])

    return df

def processing_nocat(core_df):

    mapping_asm = {'SAT': 1, 'FC': 2, 'WP':3}
    mapping_soils = {'Clay':1, 'Silty Clay':2, 'Sandy Clay':3, 'Silty Clay Loam':4,
       'Sandy Clay Loam':5, 'Loam':6, 'Silt Loam':7, 'Sandy Loam':8, 'Loamy Sand':9,
       'Sand':10}

    df=core_df.replace({'SoilConditions': mapping_asm})
    df=df.replace({'soil': mapping_soils})

    df = df.select_dtypes(exclude=['category','object'])
    return df

def split_asm(df):
    df_sat=df.loc[df['SoilConditions'] =='SAT']
    df_fc=df.loc[df['SoilConditions'] =='FC']
    df_wp=df.loc[df['SoilConditions'] =='WP']
    
    return df_sat,df_fc,df_wp

def split(df):
    feature_names = (df.drop(labels="f_var", axis=1).columns)
    X_all = df[feature_names].values
    y_all = pd.DataFrame(df['f_var']).values.ravel()

    #split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.3, shuffle=True,random_state=42)
    
    return X_all,y_all,X_train, X_test, y_train, y_test,feature_names


def feat_select(pipeline,X_all, y_all,feature_names):
    _ = StratifiedKFold(random_state=42)
    feature_selector_cv = feature_selection.RFECV(pipeline,cv=10, step=1,scoring='neg_mean_squared_error')
    feature_selector_cv.fit(X_all, y_all)
    feature_selector_cv.n_features_
    rank=feature_selector_cv.ranking_
    cv_grid_mse = -feature_selector_cv.grid_scores_

    selected_features = feature_names[feature_selector_cv.support_].tolist()
    print(selected_features)
    
    return selected_features,rank


def hyper_grid():
    
    # Number of trees in random forest
    max_depths = [int(x) for x in np.linspace(start = 1, stop = 20, num = 10)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_leaf_nodes = [int(x) for x in np.linspace(10, 110, num = 11)]
    #max_leaf_nodes=max_leaf_nodes.append(None)
    # Minimum number of samples required to split a node
    min_samples_splits = [2,3,4, 5, 6,7,8,9,10]
    # Minimum number of samples required at each leaf node
    min_samples_leafs = [1,2,3,4, 5, 6,7,8,9,10]
    # Method of selecting samples for training each tree
    
    splitters=['best','random']
    presorts=[True,False]

    min_impurity_decreases = [0,1,2,3,4, 5, 6,7,8,9,10]
    
    # Create the random grid
    random_grid = {'reg__max_depth': max_depths,
                   'reg__max_features': max_features,
                   'reg__max_leaf_nodes': max_leaf_nodes,
                   'reg__min_samples_split': min_samples_splits,
                   'reg__min_samples_leaf': min_samples_leafs,
                  'reg__min_impurity_decrease':min_impurity_decreases,
                  'reg__splitter':splitters,
                  'reg__presort':presorts}
    
    return random_grid, max_depths, max_features, max_leaf_nodes, min_samples_splits,min_samples_leafs,splitters,presorts,min_impurity_decreases
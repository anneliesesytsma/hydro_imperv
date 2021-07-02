from pyswmm import Simulation;
from pyswmm import Subcatchments;
from pyswmm import Nodes;
from pyswmm import SystemStats;
from pyswmm import Simulation;
from swmmtoolbox import swmmtoolbox;

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
from sklearn import tree
from sklearn.tree import _tree

import matplotlib.font_manager as font_manager
from matplotlib.font_manager import FontProperties

def soil_pick(ASM,soil):
    """
    Function to return soil specific ASM soil conditions
    
    input: 
    1. ASM ("FC", "WP", "SAT") 
    2. soil dataframe
    
    output:
    soil dataframe with Cond == ASM
    
    """
    
    soil_dum = soil[soil.Cond ==ASM]
    return soil_dum



def update_GA(read,write,perm_line, H_i,K,IMD):
    """
    function to update Green Ampt soil paramters in SWMM .inp file
    
    inputs:
        read: read file
        write: write file
        perm_line: line that contains permeable catchment soil parameters in the .inp file
        H_i: suction head (in)
        K: saturated hydraulic conductivity (in/hr)
        IMD: initial moisture deficit
        
     outputs:
         updates .inp file 
       
    """
    
    fin = open(read)
    fout = open(write, "wt")
    for perm_line in fin: #note that these values have to be updated if the read_file inputs change
        fout.write(perm_line.replace('S2              	12.6      	0.01      	0.097     ', 
                                'S2              	'+ str(H_i) + '      	'+ str(K) + '      	'+ str(IMD) + ''))
    fin.close()
    fout.close()


def paths(P,ASM,current_dir):
    
    """
    Function to return read, write filepaths for .inp files and outpath for csv files
    
    input: 
    1. 'write' inp file names
    2. 'read' inp file names
    
    output:
    wrote path, read path, outpath
    
    """
    read='P_'+str(P)+'in_READ.inp'
    write='P_'+str(P)+'in_WRITE.inp'
    write_path = '\\'.join([current_dir, 'inp_files\\SCSTYPEII_MGA\\'+str(write)])
    read_path = '\\'.join([current_dir, 'inp_files\\SCSTYPEII_MGA\\'+str(read)])
    out_path= '\\'.join([current_dir, 'out\\'])
    run= 'P'+str(P)+'_'+'ASM'+str(ASM)
    return write_path,read_path,out_path,run

def test_sim(write):
    """
    Function to test a single simulation.
    
    input: write path
    
    """
    with Simulation(write,'r') as sim:
        for step in sim:
            pass
        system_routing = SystemStats(sim)
        sim.report()
        print(system_routing.runoff_stats)
        
def run_sim(write,A1,A2,W1,W2,S,P):
    
    """
    run pyswmm
    
    inputs:
        write = write file path
        A1 = impervious subcatchment area
        A2 = pervious subcatchment area
        W1 = impervious subcatchment width
        W2 = pervious subcatchment width
        S = slope
        P = rainfall depth
        
    outputs:
        infiltration = total infiltration
        runoff = total runoff
        runon = total runon
    
    """

    with Simulation(write,'r') as sim:

        d=dict()
        s1 =  Subcatchments(sim)["S1"]
        s2 =  Subcatchments(sim)["S2"]
        s1.area = A1
        s2.area = A2
        s1.width = W1
        s2.width = W2
        s1.slope = S/100.
        s2.slope = S/100.
        for step in sim:
            pass
        system_routing = SystemStats(sim)
        sim.report()
        d=s2.statistics # sets dictionary keys
        infiltration=d['infiltration']
        runoff=d['runoff']
        runon=d['runon']
        
    return infiltration,runoff,runon


def boxplots(x,y):
    """
    Function to return boxplots
    
    input:
        x = rows
        y = columns
        
    output:
        f = figure
        axes = x*y 
        ab = labels
        
    """
     
    n=x*y # n is number of subplots
    f, axes = plt.subplots(x,y,sharey=True,figsize= (12,14))
    f.subplots_adjust(left=0.09, bottom=0.1, right=0.99, top=0.99, wspace=0.2,hspace = 0.7)
#     sns.set_context("paper", font_scale=1.7) 
    labels=list(string.ascii_uppercase)
    n_list=list(range(n))
    ab = zip(labels, n_list)
    for i,(a,b) in enumerate(ab):
        if (i % 2) == 0:
            f.text(0.0, 1.01*(n-0.97*i)/n, a, fontname='Arial', size=18, color='black', weight="bold", horizontalalignment='left', verticalalignment='center')              
        else:
            f.text(0.53, 1.01*(n-0.97*i)/n+0.12, a, fontname='Arial', size=18, color='black', weight="bold", horizontalalignment='left', verticalalignment='center')  

    return f, axes,ab

def bins(core):
    """
    Function to create 10 bins for continuous variables
    
    Inputs:
        core = dataframe with continuous variables fV, W_m, W_norm,A_hec
        
    Outputs:
        core dataframe with new columns for binned variables
    
    """
    
    bins=np.linspace(0, 1., 11).astype(float).tolist()
    W_bins=np.linspace((core['W_norm'].min()-1).astype(float).round(2), 
                       (core['W_norm'].max()+1).astype(float).round(2), 11).astype(float).tolist()

    W_binsm=np.linspace((core['W_m'].min()-1).astype(float).round(2), 
                       (core['W_m'].max()+1).astype(float).round(2), 11).astype(float).tolist()
    
    labels = ','.join(str(e) for e in bins)[4:].split(",")
    W_labels = ','.join(str(e) for e in W_bins)[6:].split(",")
    W_labelsm = ','.join(str(e) for e in W_binsm)[6:].split(",")
    
    core['fV_bins'] = pd.cut(x=core['fV'], bins=bins, labels=labels).astype(float).round(2) 
    core['a_bins'] = pd.cut(x=core['A_hec'], bins=bins, labels=labels).astype(float).round(2) 
    core['w_bins'] = pd.cut(x=core['W_norm'], bins=W_bins, labels=W_labels).astype(float).round(2) 
    core['w_binsm'] = pd.cut(x=core['W_m'], bins=W_binsm, labels=W_labelsm).astype(float).round(2) 
    core['w_bins2']= core['w_bins'] 
    return core

def processing_nocat(core_df):

    mapping_asm = {'SAT': 1, 'FC': 2, 'WP':3}
    mapping_soils = {'Clay':1, 'Silty Clay':2, 'Sandy Clay':3, 'Silty Clay Loam':4,
       'Sandy Clay Loam':5, 'Loam':6, 'Silt Loam':7, 'Sandy Loam':8, 'Loamy Sand':9,
       'Sand':10}

    df=core_df.replace({'ASM': mapping_asm})
    df=df.replace({'soil': mapping_soils})

    df = df.select_dtypes(exclude=['category','object'])
    return df

# def split_asm(df):
    
#     """
#     """
    
#     df_sat=df.loc[df['ASM'] =='SAT']
#     df_fc=df.loc[df['ASM'] =='FC']
#     df_wp=df.loc[df['ASM'] =='WP']
    
#     return df_sat,df_fc,df_wp

def split(df):
    """
    
    Funcion to split data into X and y training and testing sets
    
    inputs:
        df = dataframe with column 'f_var'
    
    outputs:
        X_all
        y_all
        X_train
        y_train
        X_test
        y_test
        feature names
        
    """
    
    feature_names = (df.drop(labels="f_var", axis=1).columns)
    X_all = df[feature_names].values
    y_all = pd.DataFrame(df['f_var']).values.ravel()

    #split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.3, shuffle=True,random_state=42)
    
    return X_all,y_all,X_train, X_test, y_train, y_test,feature_names


def hyper_grid():
    """
    
    Function to produce a hyperparamter grid
    
    """
    
    # Depth of tree
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

def random_state(x_all_select,y_all):
    """
    Function to test sensitivity of outcome and error metrics to the random state variable
    Tests 30 random states from 100 to 100000
    
    inputs:
        X_all_select = all of the selected X variables
        y_all = all of the y data
    
    outputs:
        mse_all_rs = dictionary of mse across different random states
        scores_all_rs = dictionary of r2 across different random states
        
    """
    # splits X, Y  input data into X and y training and testing sets
    mse_all_rs=dict()
    scores_all_rs=dict()
    randomstate=np.random.randint(100,100000,30)
    for i in randomstate:
         # splits X, Y  input data into X and y training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(x_all_select,y_all, test_size=0.30,shuffle=True, random_state=i) 

        tree=DecisionTreeRegressor(random_state=i)
        tree.fit(X_train, y_train)
        y_pred=tree.predict(X_test)
        
        score=r2_score(y_test,y_pred)
        mse=mean_squared_error(y_test,y_pred)
    
        mse_all_rs.update({i:mse})
        scores_all_rs.update({i:score})

    return mse_all_rs,scores_all_rs

def cv_10(X_train,y_train,best_reg):


    """
    10-Folds cross-validator.
    Split dataset into 10 consecutive folds.
    Each fold is then used once as a validation while the k - 1 remaining folds form the training set.
    
    intputs:
        X_train = X training data
        y_train = y training data
        best_reg = regression function
        
    outputs:
        mse_all = dictionary of all mse values across splits
        scores_all = dictionary of all r2 values across splits
    
    prints:
        mean mse
        mean r2
    """
    mse_all=dict()
    scores_all=dict()
    
    #specify regression tree model, cv
    cv = KFold(n_splits=10, random_state=42)

    #perform 10-fold cv, append 10 scores
    n=0
    for train_index, test_index in cv.split(X_train): 
        n=n+1
        # split X_train into 10 folds (10 different X_trains)
        X_train_cv, X_test_cv, y_train_cv, y_test_cv = X_train[train_index], X_train[test_index], y_train[train_index], y_train[test_index]
        # fit a decision tree regressor to the fold
        best_reg.fit(X_train_cv, y_train_cv)
        y_predicted = best_reg.predict(X_test_cv)

        cv_score=best_reg.score(X_test_cv, y_test_cv)
        mse=mean_squared_error(y_test_cv, y_predicted)
    
        mse_all.update({n:mse})
        scores_all.update({n:cv_score})
    
    mse_all=dict_to_df(mse_all,'all')
    scores_all=dict_to_df(scores_all,'all')
    print('mean cv_mse = ' + str(mse_all['value'].mean()))
    print('mean cv_r2 = ' + str(scores_all['value'].mean()))
    print('----------------')
    return mse_all,scores_all

def plot_prediction(X_train, y_train,X_test_cv,y_test_cv,ax,best_reg,mse_all,scores_all):
    """
    Function to plot actual vs. predicted y data
    
    intputs:
        X_train = X training data
        y_train = y training data
        X_test_cv = X testing data for plotting the testing errors
        y_test_cv = y testing data for plotting the testing errors 
        ax = axes for plot
        best_reg = regression function
        
    outputs:
        plot of actual vs. predicted data

    """
    if y_test_cv=="": # if no y_test_cv input, create y_test_cv 
        X_train, X_test_cv, y_train, y_test_cv = train_test_split(X_train, y_train, test_size=0.3, shuffle=True,random_state=42)
    
    #specify regression tree model, cv
    cv_predict=cross_val_predict(best_reg, X_test_cv, y_test_cv, cv=10)

    ax.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'k--', lw=2)
    ax.scatter(y_test_cv, cv_predict, color='grey',alpha = 0.1) 
    ax.set_xlabel('PySWMM Prediction',fontname='Arial', size = 16)
    ax.set_ylabel("Regression Tree Prediction",fontname='Arial', size = 16)
    ax.text(0.01,.7, r'$r^2$ = '+str(round(scores_all['value'].mean(),3))+
            "\n"            '$MSE$ = ' +str(round(mse_all['value'].mean(),4)),fontname='Arial', size = 16)
    
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontname('Arial')
        label.set_fontsize(14)  

def testing_mse(X_train, X_test, y_train, y_test,best_reg):
    """
    Function to compute the MSE and R2 of the regression on the testing data
    
    inputs:
        X_train = X training data
        X_test = X testing data
        y_train = y training data
        y_test = y testing data
        best_reg = regression function
    
    outputs:
        mse_testing =  mean square error 
        score_testing =  r2 scores
        
    """
        
    best_reg.fit(X_train, y_train)
    y_predicted = best_reg.predict(X_test)
    score_testing=best_reg.score(X_test, y_test)
    mse_testing=mean_squared_error(y_test, y_predicted)
    print('------------') 
    print('testing mse = ' + (str(mse_testing)))
    print('testing r2 = ' + (str(score_testing)))
    
    return mse_testing,score_testing

def dict_to_df(d,name):
    return pd.DataFrame(d.items(), columns=[name, 'value'])

def feature_imp(X_train, y_train,features,best_reg):
    """
    
    Function to return gini importance indices for input features using 10-fold CV.
    
    inputs:
        X_train = x training data
        y_train = y training data
        features = features to test importance of
        best_reg = regression function
    
    outputs:
        mse_all = mean square error of prediction vs. observed data
        importance_all = feature importance indices
        
    """
    importance=dict()
    mse=dict()
    
    importance_all=dict()
    mse_all=dict()

    #specify regression tree model, cv
    cv = KFold(n_splits=10, random_state=42,shuffle=True)

    #perform 10-fold cv, append 10 scores
    n=0
    for train_index, test_index in cv.split(X_train): 
        n=n+1
        # split X_train into 10 folds (10 different X_trains)
        X_train_cv, X_test_cv, y_train_cv, y_test_cv = X_train[train_index], X_train[test_index], y_train[train_index], y_train[test_index]
        # fit a decision tree regressor to the fold
        best_reg.fit(X_train_cv, y_train_cv)
        y_predicted = best_reg.predict(X_test_cv)
        
        importance=dict(zip(features, best_reg.feature_importances_))
        importance_all.update({n:importance})
        mse=mean_squared_error(y_test_cv, y_predicted)
        mse_all.update({n:mse})

    return mse_all,importance_all

def tree_to_code(tree, feature_names):

    '''
    Outputs a decision tree model as an ArcPython function

    Parameters:
    -----------
    tree: decision tree model
        The decision tree to represent as a function
    feature_names: list
        The feature names of the dataset used for building the decision tree
    
    '''

    
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    print("for row in cursor:")
    
    def recurse(node, depth):
        indent = "    " * depth
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            print("{}if ({} <= {}):".format(indent,name, threshold))
            recurse(tree_.children_left[node], depth + 1)
            print("{}elif ({} > {}):".format(indent, name, threshold))
            recurse(tree_.children_right[node], depth + 1)
        else:
            print("{}row[4]={}".format(indent, str(tree_.value[node])[2:-2]))
            print("    " * depth + "cursor.updateRow(row)".format(indent))
            
    recurse(0, 1)
#Import libraries
#from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn import svm
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import preprocessing
### Classifiers
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
# from sklearn.linear_model import RidgeClassifier, SGDClassifier
from sklearn.naive_bayes import BernoulliNB, GaussianNB
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.neural_network import MLPClassifier
#from sklearn.svm import LinearSVC, NuSVC, SVC
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
#from sklearn.linear_model import ElasticNetCV




################### Loop to run scikit learn models on all questions

def training_loop(d_frame, chosen_var):
    print(d_frame, chosen_var)
    results = []  #To store model accuracies
    cols = ['Model','Accuracy']  #Columns for results dataframe           
    #Defining feature set and target set
    X = d_frame.loc[:, d_frame.columns != '%s' % chosen_var]
    y = d_frame[['%s' % chosen_var]]
    
    
            
            
    #Split into train and test datasets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    ###########################################################################################################
    # Construct pipelines for models - Random Forest, SVM, KNeighbors, Gradient Boosting, 
    #                                  Gaussian Naive Bayes, ElasticNet Classifier
# pipe_lr = Pipeline([('scl', StandardScaler()),
# 			('clf', LogisticRegression(random_state=42))])
    
    # pipe_rf = Pipeline([('scl', StandardScaler()),
    #             ('clf', RandomForestClassifier(random_state=42))])
    
    
#     pipe_svm = Pipeline([('scl', StandardScaler()),
#                 ('clf', svm.SVC(random_state=42))])
    

#     pipe_knn = Pipeline([('scl', StandardScaler()),
#                 ('clf', KNeighborsClassifier())])
    
    
    pipe_xgb = Pipeline([('scl', StandardScaler()),
                ('clf', GradientBoostingClassifier())])
    
    
    pipe_gnb = Pipeline([('scl', StandardScaler()),
                 ('clf', GaussianNB())])
    
    pipe_DT = Pipeline([('scl', StandardScaler()),
            ('clf', DecisionTreeClassifier())])
    
#     pipe_enc = Pipeline([('scl', StandardScaler()),
#                 ('clf', ElasticNetCV())])   
                
    #Parameters for grid search - Grid search tunes hyperparameters for each model
    param_range = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    param_range_fl = [1.0, 0.5, 0.1]
    
    #grid_params_lr = [{'clf__penalty': ['l1', 'l2'],
    #		'clf__C': param_range_fl,
    #		'clf__solver': ['liblinear']}] 
    
#     grid_params_rf = [{'clf__criterion': ['gini', 'entropy']}]    	
#         #	'clf__min_samples_leaf': param_range,
#         #	'clf__max_depth': param_range,
#         #	'clf__min_samples_split': param_range[1:]}]
    
#     grid_params_svm = [{'clf__kernel': ['linear', 'rbf'], 
#             'clf__C': param_range}]
    
#     grid_params_knn = [{'clf__n_neighbors': np.arange(1, 25)}]
    
    grid_params_xgb = [{'clf__learning_rate' : [0.15,0.1,0.05,0.01,0.005,0.001]}]
                    #   'clf__n_estimators' : [200],
                    #   'clf__max_depth' : [2,3,4,5,6],
                    #   'clf__min_samples_split' : [0.005, 0.01, 0.05, 0.10],
                    #   'clf__min_samples_leaf' : [0.005, 0.01, 0.05, 0.10],
                    #   'clf__max_features' : ["auto", "sqrt", "log2"],
                    #   'clf__subsample' : [0.8, 0.9, 1]}] 
    
    grid_params_gnb = [{ 'clf__var_smoothing': [1e-9, 1e-8,1e-7, 1e-6, 1e-5]}]
    
    grid_params_DT = [{'clf__criterion' : ['gini', 'entropy'],
                    'clf__splitter': ['best', 'random'],
                    'clf__class_weight': [None, 'balanced'],
                    'clf__max_features': ['auto', 'sqrt', 'log2'],
                    'clf__max_depth' : [1,2,3, 4, 5, 6, 7, 8],
                    'clf__min_samples_split': [0.005, 0.01, 0.05, 0.10],
                    'clf__min_samples_leaf': [0.005, 0.01, 0.05, 0.10]}]

#     grid_params_enc = [{'clf__normalize' : [True, False],
#                     'clf__selection' : ['cyclic', 'random']}]
#                     #'clf__l1_ratio' : np.arange(0, 1, 0.01),
#                     #'clf__alphas': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.0, 1.0, 10.0, 100.0]}]    
    
    
    # Construct grid searches
    jobs = -1
    
# model_lr = GridSearchCV(estimator=pipe_lr,
# 			param_grid=grid_params_lr,
# 			scoring='accuracy',
# 			cv=10) 
                
    # model_rf = GridSearchCV(estimator=pipe_rf,
    #             param_grid=grid_params_rf,
    #             scoring='accuracy',
    #             cv=10, 
    #             n_jobs=jobs)
    
    
    # model_svm = GridSearchCV(estimator=pipe_svm,
    #             param_grid=grid_params_svm,
    #             scoring='accuracy',
    #             cv=10,
    #             n_jobs=jobs)
    
    
    # model_knn = GridSearchCV(estimator=pipe_knn, 
    #                     param_grid = grid_params_knn,
    #                     scoring='accuracy',
    #                     cv=10,
    #                     n_jobs = jobs)
    
    model_xgb = GridSearchCV(estimator=pipe_xgb,
                param_grid=grid_params_xgb,
                scoring='accuracy',
                cv=10) 
                
    model_gnb = GridSearchCV(estimator=pipe_gnb,
                param_grid=grid_params_gnb,
                scoring='accuracy',
                cv=10, 
                n_jobs=jobs)
    
    
    model_DT = GridSearchCV(estimator=pipe_DT,
                param_grid=grid_params_DT,
                scoring='accuracy',
                cv=10,
                n_jobs=jobs)
    
    # model_enc = GridSearchCV(estimator=pipe_enc,
    #             param_grid=grid_params_enc,
    #             scoring='r2',
    #             cv=10,
    #             n_jobs=jobs)
    
    # List of all pipelines built
    #grids = [ model_rf,model_svm, model_knn,model_xgb, model_gnb,model_enc]
    grids = [ model_DT, model_gnb, model_xgb]
    
    # grid_dict = {#0: 'Logistic Regression', 
    #             0: 'Decision Trees', 
    #         #1: 'Support Vector Machine', 2: 'K-Nearest Neighbors', 
    #         #3: 'Gradient Boosting',
    #         4: 'Gaussian Naive Bayes' #,  5: 'Elastic-Net Classifier'}
    # }

    grid_dict = { 
                0: 'Decision Trees', 
            1: 'Gaussian Naive Bayes', 
            2: 'Gradient Boosting'
    }
    
    #Fitting data and calculate accuracy for each model pipeline
    print('Performing model optimizations for')
    best_acc = 0.0
    best_clf = 0
    
    for idx, gs in enumerate(grids):
        print('\nEstimator: %s' % grid_dict[idx])	
        # Fit grid search	
        gs.fit(X_train, y_train.values.ravel())
        # Best params
        print('Best params: %s' % gs.best_params_)
        # Best training data accuracy
        print('Best training accuracy: %.3f' % gs.best_score_)
        # Predict on test data with best params
        y_pred = gs.predict(X_test)
        # Test data accuracy of model with best params
        print(accuracy_score(y_test, y_pred))
        #print('Test set accuracy score for best params: %.3f ' % accuracy_score(y_test, y_pred.round()))
        # Find out best (highest test accuracy) model
        if accuracy_score(y_test, y_pred) > best_acc:
            best_acc = accuracy_score(y_test, y_pred)
            best_clf = idx
            best_parameters = pd.DataFrame(gs.best_params_.items(), columns=['Parameter', 'Value'])
        results.append([grid_dict[idx], gs.best_score_])
        
        #best_parameters.append(best_parameters)
    print('\nClassifier with best test set accuracy: %s' % grid_dict[best_clf])
    

    #results.append([grid_dict, gs.best_score_])
    
    # Save best grid search pipeline to file
# dump_file = 'best_gs_pipeline_{}.pkl'.format(name)
    # dump_file = 'Model_{}.pkl'.format('DT')
    # joblib.dump(best_gs, dump_file, compress=1)         
    # print('\nSaved %s grid search pipeline to file: %s' % (grid_dict[best_clf], dump_file))

    #Convert results list into dataframe
    results =  pd.DataFrame(results, columns=cols).sort_values(by='Accuracy', ascending=False)
    #cols_params = ['Parameter', 'Value']
    best_params = best_parameters
    classifier_best_test = grid_dict[best_clf]
    return results, best_params, classifier_best_test
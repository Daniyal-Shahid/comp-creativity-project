# %%
import pandas as pd

# plot imports
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.figure_factory as ff
import plotly.express as px

# model imports
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# imports for model saving
import joblib
import os
import json

# flag for github plotting
github = True

try: 
    os.mkdir('images')
except:
    print("Folder already exists")

# %%
data = pd.read_csv('data.csv')
data.info()

# %%
# unique labels
data['label'].unique()

# %%
# train/validation/test split
training, test = train_test_split(data, stratify=data['label'], train_size=0.8, random_state=1)

# brief description of data
training.describe()

# %%

features = ['tempo', 'beats', 'chroma_stft', 'rmse',
            'spectral_centroid', 'spectral_bandwidth', 'rolloff', 'zero_crossing_rate']

# plotting of 4 first variables
fig = make_subplots(cols=4, rows=2, subplot_titles=features)

for i, feature in enumerate(features):
    fig.add_trace(go.Histogram(x=training[feature], name=feature), row=i//4 + 1, col=i%4 + 1)

width = 1000
height = 600

fig.update_layout(width=width, height=height, title='feature distribution')

if github:
    fig.show('png', width=width, height=height)
else:
    fig.show()

fig.write_image(os.path.join('images', fig.layout['title']['text'] + '.png'))


# %%
# heatmap of correlation between features
fig = px.imshow(training[features].corr())

fig.update_layout(title='feature correlation heatmap')

if github:
    fig.show('png')
else:
    fig.show()

fig.write_image(os.path.join('images', fig.layout['title']['text'] + '.png'))


# %% [markdown]
# As can be seen above, there are several features that are highly correlated to each other. `spectral_bandwidth`, `rolloff` and `zero_crossing_rate` all share high correlation with `spectral_centroid`, and `beats` is highly correlated with `tempo`; some of these features will therefore be removed.

# %%
# drop spectral_bandwidth, rolloff, zero_crossing_rate and beats
training.drop(columns=['spectral_bandwidth', 'rolloff', 'zero_crossing_rate', 'beats'], inplace=True)
test.drop(columns=['spectral_bandwidth', 'rolloff', 'zero_crossing_rate', 'beats'], inplace=True)


# %% [markdown]
# # Feature exploring
# Now, we will investigate the features we're going to be working on. For that purpose, we'll plot the kernel density estimation across different target classes, as well as give a brief description of each feature.
# 
# It is important to note that the vertical axis on the plots represents the density of the variable being plotted; this means that the scale will be influenced by the range of each variable.
# 
# ## Tempo
# The `tempo` feature tells us the rhythm of a song. The higher the value, the fastest the song plays.

# %%
genres = training['label'].unique()

feature='tempo'
fig = ff.create_distplot(
    hist_data=[training[feature].loc[training['label'] == genre] for genre in genres],
    group_labels=genres, show_hist=False, show_rug=False
)

width = 1000

fig.update_layout(width=width, title=f'{feature} distribution')

if github:
    fig.show('png', width=width)
else:
    fig.show()

fig.write_image(os.path.join('images', fig.layout['title']['text'] + '.png'))

# %% [markdown]
# ## chroma_stft
# The `chroma_stft` feature represents the [Short-time Fourier transform](https://en.wikipedia.org/wiki/Short-time_Fourier_transform) of each song. This feature is however reduced to a single value, and the dataset doesn't include an explanation of what it represents (the complete Fourier transform would have a single value for every frequency, representing the contribution of that frequency to the entirety of the song). Given the range of values present in the dataset, we can assume they correspond to the proportion of the dominant frequency in each song.

# %%
feature='chroma_stft'
fig = ff.create_distplot(
    hist_data=[training[feature].loc[training['label'] == genre] for genre in genres],
    group_labels=genres, show_hist=False, show_rug=False
)

width = 1000

fig.update_layout(width=width, title=f'{feature} distribution')

if github:
    fig.show('png', width=width)
else:
    fig.show()

fig.write_image(os.path.join('images', fig.layout['title']['text'] + '.png'))

# %% [markdown]
# ## rmse
# 
# `rmse` stands for [root mean square energy](https://musicinformationretrieval.com/energy.html#:~:text=The%20root%2Dmean%2Dsquare%20energy,x%2C%20sr%20%3D%20librosa) and represents the total amount of energy in a given signal.

# %%
feature='rmse'
fig = ff.create_distplot(
    hist_data=[training[feature].loc[training['label'] == genre] for genre in genres],
    group_labels=genres, show_hist=False, show_rug=False
)

width = 1000

fig.update_layout(width=width, title=f'{feature} distribution')

if github:
    fig.show('png', width=width)
else:
    fig.show()

fig.write_image(os.path.join('images', fig.layout['title']['text'] + '.png'))

# %% [markdown]
# ## spectral_centroid
# 
# The `spectral_centroid` [property](https://en.wikipedia.org/wiki/Spectral_centroid) indicates the frequency that corresponds to the center of mass of a signal.

# %%
feature='spectral_centroid'
fig = ff.create_distplot(
    hist_data=[training[feature].loc[training['label'] == genre] for genre in genres],
    group_labels=genres, show_hist=False, show_rug=False
)

width = 1000

fig.update_layout(width=width, title=f'{feature} distribution')

if github:
    fig.show('png', width=width)
else:
    fig.show()

fig.write_image(os.path.join('images', fig.layout['title']['text'] + '.png'))

# %% [markdown]
# The plots above show us that the features `tempo`, `chroma_stft`, `rmse` and `spectral_centroid` all have significant differences in distriburion across the different target classes; thus, they are going to be useful in helping us build models to predict music genre.

# %% [markdown]
# ## MFCC coefficients
# 
# MFCCs stands for [mel-frequency cepstral coefficients](https://en.wikipedia.org/wiki/Mel-frequency_cepstrum) and collectively make up an MFC (mel-frequency cepstrum). In our dataset there are 20 MFCC's. The correlation between each pair will be plotted bellow.

# %%
mfcc = [f'mfcc{x}' for x in range(1, 21)]

# heatmap of correlation between MFCC indexes
fig = px.imshow(training[mfcc].corr(), width=700, height=600)

width = 700
height = 600

fig.update_layout(width=width, height=height, title='MFCC index correlation heatmap')

if github:
    fig.show('png', width=width, height=height)
else:
    fig.show()

fig.write_image(os.path.join('images', fig.layout['title']['text'] + '.png'))

# %% [markdown]
# By observing the plot above we can see a high correlation in groups of odd and even coefficients, excluding the first and second MFCC's. To reduce the dimensionality of this feature, we will apply the PCA method to reduce the numbers of both odd and even MFCC's.

# %%
odd_mfcc = [f'mfcc{x}' for x in range(3, 21) if (x % 2 == 1)]
even_mfcc = [f'mfcc{x}' for x in range(3, 21) if (x % 2 == 0)]

# pca transformers
odd_pca = PCA().fit(training[odd_mfcc])
even_pca = PCA().fit(training[even_mfcc])

# pca transformation on training dataset
training_pca_odd_components = odd_pca.transform(training[odd_mfcc])
training_pca_even_components = even_pca.transform(training[even_mfcc])

# pca transformation on test dataset
test_pca_odd_components = odd_pca.transform(test[odd_mfcc])
test_pca_even_components = even_pca.transform(test[even_mfcc])

print(f"Explained variance (on tranining set) of reduced odd MFCC's:")
print(odd_pca.explained_variance_ratio_)
print('\n')

print(f"Explained variance (on tranining set) of reduced even MFCC's:")
print(even_pca.explained_variance_ratio_)

# %% [markdown]
# Following the results above, we can conclude that the majority (roughly 97%) of the information found in groups of odd and even coefficients can be synthesized in the first 3 components; we will keep, however, 7 components from each group, reducing the total number of dimensions from 18 to 14 in order to keep 99% of the information from the features.

# %%
n_pca_components = 7

training[[f'reduced_odd_mfcc_{i+1}' for i in range(n_pca_components)]] = training_pca_odd_components[:, :n_pca_components]
training[[f'reduced_even_mfcc_{i+1}' for i in range(n_pca_components)]] = training_pca_even_components[:, :n_pca_components]

test[[f'reduced_odd_mfcc_{i+1}' for i in range(n_pca_components)]] = test_pca_odd_components[:, :n_pca_components]
test[[f'reduced_even_mfcc_{i+1}' for i in range(n_pca_components)]] = test_pca_even_components[:, :n_pca_components]



# %% [markdown]
# ## Exploring newly created features
# 
# Now we will further explore our new features, as well as the original `mfcc1` and `mfcc2`, by plotting their distribution across the dataset and also across the different labels.

# %%
to_plot = ['mfcc1', 'mfcc2']
for i in range(n_pca_components):
    to_plot.extend([f'reduced_odd_mfcc_{i+1}', f'reduced_even_mfcc_{i+1}'])

# plotting of 4 first variables
fig = make_subplots(cols=2, rows=round(len(to_plot)/2), subplot_titles=to_plot)

for i, feature in enumerate(to_plot):
    fig.add_trace(go.Histogram(x=training[feature], name=feature), row=int(i/2)+1, col=(i%2)+1)

width = 1000
height = 300*round(len(to_plot)/2)

fig.update_layout(width=width, height=height, title='MFCC index distribution')

if github:
    fig.show('png', width=width, height=height)
else:
    fig.show()

fig.write_image(os.path.join('images', fig.layout['title']['text'] + '.png'))


# %%
for feature in to_plot:
    fig = ff.create_distplot(
        hist_data=[training[feature].loc[training['label'] == genre] for genre in genres],
        group_labels=genres, show_hist=False, show_rug=False
    )
    
    width = 1000
    height = 400

    fig.update_layout(width=width, height=height, title=f'{feature} distribution')

    if github:
        fig.show('png', width=width, height=height)
    else:
        fig.show()

    fig.write_image(os.path.join('images', fig.layout['title']['text'] + '.png'))

# %% [markdown]
# ## Standardization of features
# 
# Now, we will standardize the features (including those we created above) so they can have no mean and unit standard deviation.

# %%
# fitting of scaler

features = ['tempo', 'chroma_stft', 'rmse', 'spectral_centroid', 'mfcc1', 'mfcc2'] + [f'reduced_odd_mfcc_{i+1}' for i in range(n_pca_components)] + [f'reduced_even_mfcc_{i+1}' for i in range(n_pca_components)]

scaler = StandardScaler()
scaler.fit(training[features].values)


training[features] = scaler.transform(training[features].values)
test[features] = scaler.transform(test[features].values)

# %% [markdown]
# ## Fitting a Naive Bayes model
# 
# In order to have a baseline for our predictive models, we will fit a simple gaussian naive bayes model and see how it performs. After that, we will train different kinds of models and compare them.

# %%
from sklearn.naive_bayes import GaussianNB

# fitting the model with the training data
gnb = GaussianNB()

gnb_scores = cross_val_score(gnb, training[features], training['label'], cv=5)

print(f'The Naive Bayes model achieved an average accuracy of {gnb_scores.mean()}.')

# %% [markdown]
# As can be seen above, the model achieved around 50% accuracy.
# 
# ## Fitting a logistic regression model

# %%
from sklearn.linear_model import LogisticRegression

# fitting the linear regression model
mlr = LogisticRegression(multi_class='ovr')

mlr_scores = cross_val_score(mlr, training[features], training['label'], cv=5)

print(f'The logistic regression model achieved an average accuracy of {mlr_scores.mean()}.')

# %% [markdown]
# ## Fitting a K-nearest neighbors classifier

# %%
from sklearn.neighbors import KNeighborsClassifier

knc = KNeighborsClassifier()

knc_scores = cross_val_score(knc, training[features], training['label'], cv=5)

print(f'The K-nearest neighbors model achieved an average accuracy of {knc_scores.mean()}.')

# %% [markdown]
# ## Fitting a Decision Tree model

# %%
from sklearn.tree import DecisionTreeClassifier

dtc = DecisionTreeClassifier()

dtc_scores = cross_val_score(dtc, training[features], training['label'], cv=5)

print(f'The Decision Tree classifier achieved an average accuracy of {dtc_scores.mean()}.')

# %% [markdown]
# ## Fitting a Random Forest Model

# %%
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier()

rfc_scores = cross_val_score(rfc, training[features], training['label'], cv=5)

print(f'The Random Forest classifier achieved an average accuracy of {rfc_scores.mean()}.')

# %% [markdown]
# ## Fitting a Support Vector Classifier

# %%
from sklearn.svm import SVC

svc = SVC()

svc_scores = cross_val_score(svc, training[features], training['label'], cv=5)

print(f'The Support Vector classifier achieved an average accuracy of {svc_scores.mean()}.')

# %% [markdown]
# ## Fitting an Gradient Boosting Classifier

# %%
from xgboost import XGBClassifier

xgbc = XGBClassifier(verbosity=0)

xgbc_scores = cross_val_score(xgbc, training[features], training['label'], cv=5)

print(f'The Extreme Gradient Boosting classifier achieved an average accuracy of {xgbc_scores.mean()}.')

# %% [markdown]
# ## Fitting a Neural Neural Network classifier

# %%
from sklearn.neural_network import MLPClassifier

nnc = MLPClassifier(random_state=1)

nnc_scores = cross_val_score(nnc, training[features], training['label'], cv=5)

print(f'The Neural Network classifier achieved an average accuracy of {nnc_scores.mean()}.')

# %% [markdown]
# Of all the models trained, we will fine-tune only the 4 with the best score: the random forest, the support vector, the gradient boosting and the neural network classifiers.

# %%
# creation of folder that will contain the models
try:
    os.mkdir('models')
except:
    print('Folder already exists')

# %%
# wide search for random forest classifier

rfc = RandomForestClassifier(random_state=1)

param_grid = {
    'n_estimators': [20, 50, 100, 150, 200, 300, 500, 700, 1000],
    'max_depth': [None, 2, 5, 10, 15, 20, 30],
    'min_samples_split': [1, 2, 4, 5, 10],
    'min_samples_leaf': [1, 2, 3, 5],
    'max_features': ['sqrt', 'log2', None]
}

random_search_rfc = RandomizedSearchCV(rfc, param_grid, n_iter=64, scoring='accuracy', cv=5, n_jobs=-1, verbose=1, random_state=1)

random_search_rfc.fit(training[features], training['label'])

# %%
print("Best hyperparameters found in wide randomized search for Random Forest Classifier:")
print(random_search_rfc.best_params_)
print('\n')
print('Best accuracy:')
print(random_search_rfc.best_score_)

# %%
# narrow search for random forest classifier

rfc = RandomForestClassifier(random_state=1)

param_grid = {
    'n_estimators': [1000, 1100, 1200, 1300, 1400, 1500, 2000],
    'max_depth': [None],
    'min_samples_split': [2, 4],
    'min_samples_leaf': [1],
    'max_features': ['sqrt']
}

grid_search_rfc = GridSearchCV(rfc, param_grid, scoring='accuracy', cv=5, n_jobs=-1, verbose=1)

grid_search_rfc.fit(training[features], training['label'])

# %%
print("Best hyperparameters found in narrow grid search for Random Forest Classifier:")
print(grid_search_rfc.best_params_)
print('\n')
print('Best accuracy:')
print(grid_search_rfc.best_score_)

# %%
# saves the best model
folder = 'models/random_forest_classifier'
try:
    os.mkdir(folder)
except:
    print('Folder already exists')

# save best random forest model
joblib.dump(grid_search_rfc.best_estimator_, os.path.join(folder, 'model.sav'))

with open(os.path.join(folder, 'config.json'), 'w') as json_file:

    # save best parameters and best score in json file
    json.dump(
        {
            'best_paramters': grid_search_rfc.best_params_,
            'best_score': grid_search_rfc.best_score_
        },
        json_file, 
        indent=4
    )
  

# %%
# wide search for support vector classifier

svc = SVC(probability = True)
param_grid = {
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'C': [1, 2, 5, 10, 50, 100, 1000],
    'degree': [1, 2, 3, 4, 5, 6],
    'gamma': ['scale', 'auto', 0.1, 0.5, 1, 1.5, 2, 3, 4, 5]
}
random_search_svc = RandomizedSearchCV(svc, param_grid, n_iter=128, scoring='accuracy', cv=5, n_jobs=-1, verbose=1, random_state=1)
random_search_svc.fit(training[features], training['label'])

# %%
print("Best hyperparameters found in wide randomized search for Support Vector Classifier:")
print(random_search_svc.best_params_)
print('\n')
print('Best accuracy:')
print(random_search_svc.best_score_)

# %%
# narrow search for support vector classifier

svc = SVC(probability = True)
param_grid = {
    'kernel': ['rbf'],
    'C': [4, 5, 6],
    'degree': [3, 4, 5],
    'gamma': ['scale', 'auto', 0.01, 0.1, 0.2]
}

grid_search_svc = GridSearchCV(svc, param_grid, scoring='accuracy', cv=5, n_jobs=-1, verbose=1)
grid_search_svc.fit(training[features], training['label'])

# %%
print("Best hyperparameters found in narrow grid search for Support Vector Classifier:")
print(grid_search_svc.best_params_)
print('\n')
print('Best accuracy:')
print(grid_search_svc.best_score_)

# %%
# saves the best model
folder = 'models/support_vector_classifier'
try:
    os.mkdir(folder)
except:
    print('Folder already exists')

# save best random forest model
joblib.dump(grid_search_svc.best_estimator_, os.path.join(folder, 'model.sav'))

with open(os.path.join(folder, 'config.json'), 'w') as json_file:

    # save best parameters and best score in json file
    json.dump(
        {
            'best_paramters': grid_search_svc.best_params_,
            'best_score': grid_search_svc.best_score_
        },
        json_file, 
        indent=4
    )

# %%
# wide search for extreme boosting gradient classifier

xgbc = XGBClassifier(verbosity=0, random_state=1)

param_grid = {
    'n_estimators': [100, 200, 500, 600 ,700, 800, 1000],
    'max_depth': [None, 2, 3, 5, 10, 15, 20, 30],
    'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.5, 0.7],
    'gamma': [0, 0.1, 1, 10, 100],
    'min_child_weight': [0, 0.1, 0.5, 1, 10, 100],
    'col_sample_bytree': [0.2, 0.4, 0.6, 0.8, 1],
    'reg_alpha': [0, 0.1, 1],
    'reg_lambda': [1, 1.5, 2]
}

random_search_xgbc = RandomizedSearchCV(xgbc, param_grid, n_iter=64, scoring='accuracy', cv=5, n_jobs=-1, verbose=1, random_state=1)
random_search_xgbc.fit(training[features], training['label'])

# %%
print("Best hyperparameters found in wide radomized search for Gradient Boosting Classifier:")
print(random_search_xgbc.best_params_)
print('\n')
print('Best accuracy:')
print(random_search_xgbc.best_score_)

# %%
# narrow search for extreme boosting gradient

xgbc = XGBClassifier(verbosity=0, random_state=1)

param_grid = {
    'n_estimators': [150, 200, 250, 300],
    'max_depth': [3],
    'learning_rate': [0.1],
    'gamma': [0, 0.05, 0.1],
    'min_child_weight': [0.05, 0.1, 0.2, 0.3],
    'col_sample_bytree': [0.5, 0.6, 0.7],
    'reg_alpha': [0],
    'reg_lambda': [1.5]
}

grid_search_xgbc = GridSearchCV(xgbc, param_grid, scoring='accuracy', cv=5, n_jobs=-1, verbose=1)
grid_search_xgbc.fit(training[features], training['label'])

# %%
print("Best hyperparameters found in narrow grid search for Gradient Boosting Classifier:")
print(grid_search_xgbc.best_params_)
print('\n')
print('Best accuracy:')
print(grid_search_xgbc.best_score_)

# %%
# saves the best model
folder = 'models/gradient_boosting_classifier'
try:
    os.mkdir(folder)
except:
    print('Folder already exists')

# save best random forest model
joblib.dump(grid_search_xgbc.best_estimator_, os.path.join(folder, 'model.sav'))

with open(os.path.join(folder, 'config.json'), 'w') as json_file:

    # save best parameters and best score in json file
    json.dump(
        {
            'best_paramters': grid_search_xgbc.best_params_,
            'best_score': grid_search_xgbc.best_score_
        },
        json_file, 
        indent=4
    )

# %%
# wide search for neural network

nnc = MLPClassifier(random_state=1, verbose=True)

param_grid = {
    'hidden_layer_sizes': [(10), (15), (20), (50), (100), (10, 10), (15, 15), (20, 20), (100, 100)],
    'activation': ['logistic', 'tanh', 'relu'],
    'solver': ['lbfgs', 'sgd', 'adam'],
    'alpha': [0.0001, 0.001, 0.01],
    'batch_size': ['auto', 16, 32, 64, 128],
    'learning_rate': ['constant'],  # set to constant because it is only used when sgd is the solver
    'learning_rate_init': [0.0001, 0.001, 0.01, 0.1],
    'max_iter': [100, 200, 300],
    'beta_1': [0.5, 0.7, 0.8, 0.9, 0.99],
    'beta_2': [0.8, 0.9, 0.95, 0.999]
}

random_search_nnc = RandomizedSearchCV(nnc, param_grid, n_iter=512, scoring='accuracy', cv=5, n_jobs=-1, verbose=1, random_state=1)
random_search_nnc.fit(training[features], training['label'])

# %%
print("Best hyperparameters found in wide randomized search for Neural Network Classifier:")
print(random_search_nnc.best_params_)
print('\n')
print('Best accuracy:')
print(random_search_nnc.best_score_)

# %%
# narrow search for neural network

nnc = MLPClassifier(random_state=1, verbose=True)

param_grid = {
    'hidden_layer_sizes': [(100, 100), (125, 125), (150, 150), (200, 200)],
    'activation': ['tanh'],
    'solver': ['lbfgs'],
    'alpha': [0.01, 0.02, 0.05, 0.1],
    'batch_size': [128, 160],
    'learning_rate': ['constant'],  # set to constant because it is only used when sgd is the solver
    'learning_rate_init': [0.0005, 0.001, 0.002, 0.005],
    'max_iter': [150, 200, 250],
    'beta_1': [0.75, 0.8, 0.85],
    'beta_2': [0.999]
}

grid_search_nnc = GridSearchCV(nnc, param_grid, scoring='accuracy', cv=5, n_jobs=-1, verbose=1)
grid_search_nnc.fit(training[features], training['label'])

# %%
print("Best hyperparameters found in narrow grid search for Neural Network Classifier:")
print(grid_search_nnc.best_params_)
print('\n')
print('Best accuracy:')
print(grid_search_nnc.best_score_)

# %%
# saves the best model
folder = 'models/neural_network_classifier'
try:
    os.mkdir(folder)
except:
    print('Folder already exists')

# save best random forest model
joblib.dump(grid_search_nnc.best_estimator_, os.path.join(folder, 'model.sav'))

with open(os.path.join(folder, 'config.json'), 'w') as json_file:

    # save best parameters and best score in json file
    json.dump(
        {
            'best_paramters': grid_search_nnc.best_params_,
            'best_score': grid_search_nnc.best_score_
        },
        json_file, 
        indent=4
    )

# %% [markdown]
# ## Final Results
# Here are the best mean scores using 5-fold cross validation after fine tunning all four models:
# 
# | Model | Score|
# |:------|:-----|
# | Random Forest Classifier    |    0.64125   |
# | **Support Vector Classifier**    |    **0.655**   |
# | Gradient Boosting Classifier    |    0.64   |
# | Neural Network Classifier    |    0.64875   |
# 
# The table above shows us that all the algorithms achieved similar scores, with the Support Vector Machine delivering slightly better results; it will be chosen as the final model for predicting music genre and evaluated using the test set.
# 

# %%
folder = 'models/support_vector_classifier'

model = joblib.load(os.path.join(folder, 'model.sav'))

final_score = model.score(test[features], test['label'])

print(f'The model achieved an accuracy of {final_score*100}% on the test set.')

# %% [markdown]
# Conclusions
# 
# A score of 68.5% was achieved with the chosen support vector classifier model; it is pretty impressive for such an abstract dataset with only 1000 samples to be able to achieve almost 70% of accuracy in predicting music genre from 10 different classes.



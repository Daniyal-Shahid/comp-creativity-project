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
    
# Second cell

data = pd.read_csv('data.csv')
data.info()

# Third cell
# unique labels
data['label'].unique()

# Fourth cell
# train/validation/test split
training, test = train_test_split(data, stratify=data['label'], train_size=0.8, random_state=1)

# brief description of data
training.describe()

# FIFTH CELL

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

# Sixth cell

# heatmap of correlation between features
fig = px.imshow(training[features].corr())

fig.update_layout(title='feature correlation heatmap')

if github:
    fig.show('png')
else:
    fig.show()

fig.write_image(os.path.join('images', fig.layout['title']['text'] + '.png'))


# Seventh cell

# drop spectral_bandwidth, rolloff, zero_crossing_rate and beats
training.drop(columns=['spectral_bandwidth', 'rolloff', 'zero_crossing_rate', 'beats'], inplace=True)
test.drop(columns=['spectral_bandwidth', 'rolloff', 'zero_crossing_rate', 'beats'], inplace=True)
import streamlit as st
#import numpy as np
import pandas as pd
import lightfm
import nmslib
import pickle
import scipy.sparse as sparse


def nearest_item_nms(item_id, index, n=5):
    # function for nearest neighbour search, returns builts index
    nn = index.knnQuery(item_embeddings[item_id], k=n)
    return nn

def get_title(index):
    # return titles of the item with the index "index"
    # take: idx of books
    # return - list of titles
    titles = []
    for idx in index:
        titles.append(items[items['itemid'] == idx]['title'].values[0])
    return titles


def load_embeddings(folder_name=''):
    # function for loading of embeddings from file

    with open('items_embeddings.pkl', 'rb') as f:
        item_embeddings = pickle.load(f)

    # we use nmslib for create our fast KNN
    nms_idx = nmslib.init(method='hnsw', space='cosinesimil')
    nms_idx.addDataPointBatch(item_embeddings)
    nms_idx.createIndex(print_progress=True)
    return item_embeddings,nms_idx
# load dataframe with items features

def read_files(folder_name=''):
    items = pd.read_csv(folder_name + 'items.csv')
  #  train_raitings = pd.read_csv(folder_name + 'data\\train.csv')
    return items

#folder_name = 'C:\\Users\\m_anu\\WebstormProjects\\SkillFactory\\RDS_4_Recommendation_Challenge'
folder_name = ''
items = read_files()
index = [5094, 8327, 31670]
print(get_title(index))
print('-------------')
print(items.sample(2))

item_embeddings,nms_idx = load_embeddings()

# check: try to print reccomendation for item = 100
ind_for_recom = nearest_item_nms(100, nms_idx, 4)
print(get_title(ind_for_recom[0])[1:])
print('=========================')

# the form for enter text:
itemid_for_reccomend = st.text_input('Item ID', '')
if not itemid_for_reccomend.isdigit():
    itemid_for_reccomend = 12345
    'You can start with ', itemid_for_reccomend, ' (0 - 41301)'
itemid_list = list(range(int(itemid_for_reccomend) - 5, int(itemid_for_reccomend) + 5))
# search the title of the requested item
output = items[items.itemid.isin(itemid_list)]

# select item from the list
option = st.selectbox('This item?', output['title'].values)

# ptint the title of the item
'You selected', option

# search recommendations
val_index = output[output['title'].values == option].itemid
index = nearest_item_nms(val_index, nms_idx, 7)

'Most similar items are: '
st.write('', get_title(index[0])[1:])

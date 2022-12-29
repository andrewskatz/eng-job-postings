# -*- coding: utf-8 -*-
"""
Created on Thu Sep 16 09:09:27 2021

@author: akatz4
"""



from personal_utilities import embed_cluster as ec


import pickle
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import seaborn as sns
import umap
#import umap.plot
import hdbscan
import spacy
from spacy.lang.en import English

from sklearn.manifold import MDS, TSNE


import pickle

from keybert import KeyBERT


"""

Import data and preprocess

"""


os.getcwd()

# for jee df with abstracts and full article information
proj_path = "G:/My Drive/AK Faculty/Research/Projects/project political economy of engineering education/project engineering jobs"

os.chdir(proj_path)
os.listdir()


civ_ed_df = pd.read_csv("civ_ed_grad_df_20210916.csv")





# =============================================================================
# Utility functions
# =============================================================================



def select_and_filter(original_df, text_column):
    """
    Parameters
    ----------
    item : str
        Name of the column to filter based on.
    original_df : dataframe
        Original input dataframe with text columns.

    Returns
    -------
    filtered_df : dataframe
        Dataframe that has removed the NAs in the {item} column.
    item_list : list
        List of the text in the {item} column. This list is passed to the transformer model.
    """
    original_df['added_id'] = original_df.index + 1
    filtered_df = original_df.loc[pd.notnull(original_df[text_column])]
    print(f"Filtered dataframe for {text_column}.")
    print(f"Dataframe has size {filtered_df.shape}.")
    
    item_list = filtered_df[text_column].to_list()
    print(f"The list has length {len(item_list)}.")
        
    return filtered_df, item_list

civ_ed_df.columns

civ_filtered_df, civ_filtered_list = select_and_filter(civ_ed_df, 'description')
#5637x9







def sentence_segmenter(data_frame, text_column):
    # first, create list from series of text
    
    entry_list = data_frame[text_column].to_list()
    
    # try using dictionary of lists
    
    new_df_dict = {'text_column': [],
                   'original_id': [],
                   'original_entry': [],
                   'split_sent': [],
                   'sent_num': []}
    
    nlp = English()
    # for old version of spacy
    #nlp.add_pipe(nlp.create_pipe('sentencizer'))
    # for new version of spacy
    nlp.add_pipe('sentencizer')
    
    ## using the process of converting series to list and then iterating over list
    for i, entry in enumerate(entry_list):
        doc = nlp(entry)
        # for old version of spacy
        #sentences = [sent.string.strip() for sent in doc.sents]
        # for new version of spacy
        sentences = [sent.text.strip() for sent in doc.sents]
        
        for j, sent in enumerate(sentences):
            new_df_dict['text_column'].append(text_column)
            new_df_dict['original_id'].append(i)
            new_df_dict['original_entry'].append(entry)
            new_df_dict['split_sent'].append(sent)
            new_df_dict['sent_num'].append(j)
            
    sentence_df = pd.DataFrame(new_df_dict)
    
    item_list = sentence_df['split_sent'].to_list()
    
    print(f"Filtered dataframe for {text_column}.")
    print(f"Dataframe has size {sentence_df.shape}.")
    print(f"The list has length {len(item_list)}.")

    
    return sentence_df, item_list
    





# =============================================================================
# Start processing the catme data
# =============================================================================


# First, parse on sentences in each award

civ_sentence_df, civ_sent_list = sentence_segmenter(civ_filtered_df, 'description')

civ_sentence_df.shape # 173958 x 5 (new spacy)


# pick subset dataframe of size n
size_n = 10000


civ_size_n_df = civ_sentence_df.head(size_n)

# Next, use the sentence list for embedding

#try 10000 sentences just to see
sent_embedding = ec.embed_raw_text(civ_sent_list[:size_n], 'roberta', max_seq_length=200)
    

# pickle the embeddings since there are 118175 sentences in the full 53k observation dataset
#pickle_out = open('catme_filtered_sentence_mpnet_20210829.pickle', 'wb')
pickle_out = open('civ_filtered_sentence_roberta_20210914.pickle', 'wb')
pickle.dump(sent_embedding, pickle_out)
pickle_out.close()



pickle_in = open('civ_filtered_sentence_roberta_20210914.pickle', 'rb')
sent_embedding = pickle.load(pickle_in)
pickle_in.close()




embed_param_dict = embed_param_dict= {'pca_dim': 100,
                   'n_neighbors': 4,
                   'min_dist': 0.0,
                   'n_components': 5,
                   'metric': 'cosine',
                   'random_state': 123}

embed_param_title = "pca_dim:" + str(embed_param_dict['pca_dim']) + ', n_nei:' + str(embed_param_dict['n_neighbors'])

lower_embed = ec.project_original_embedding(sent_embedding, 
                                            embed_param_dict, 
                                            to_low = False, 
                                            mid_to_low_method='umap',
                                            title=embed_param_title)


lower_embed.shape   
# print("Currently clustering item:", item)
## For hdbscan: specify min_cluster_size, min_samples, alpha
## for agglomerative specify: n_clusters, linkage
## for kmeans specify: num_clusters
cluster_param_dict = {'min_cluster_size': 5, # hdbscan options
                      'min_samples': 1,
                      'cluster_selection_epsilon': 2.0,
                      'alpha': 1.0,
                      'metric': 'euclidean',
                      'agg_type': "threshold", # agglomerative options - can be "threshold"or "n_cluster"
                      'n_clusters': 60,
                      'threshold_val': 80,
                      'affinity': 'euclidean',
                      'linkage': 'ward',
                      'num_clusters': 30}

# version to use the lower-dimensional embedding for the clustering     
cluster_res = ec.cluster_embedding(data=lower_embed, original_corpus_list=civ_sent_list[:size_n], 
                  model='agglomerative', param_dict=cluster_param_dict, plot_option=True)


all_cluster_labels = cluster_res.labels_


civ_sentence_df['cluster_label'] = all_cluster_labels

# if using a subset of the data
civ_size_n_df['cluster_label'] = all_cluster_labels

civ_size_n_df.shape
civ_size_n_df.columns




sent_embed_df = pd.DataFrame(sent_embedding)
sent_embed_df.columns


jee_joined_df = pd.concat([jee_sentence_df.reset_index(drop = True), sent_embed_df.reset_index(drop=True)], axis = 1)












"""

Summarizing using keyBERT

"""








# extract keywords for each cluster in Q5.2

def cluster_keybert(clustered_df, ngram_low = 1, ngram_high = 3):
    
    item_cl_kw_dict = {'cluster': [],
                   'cl_sentence_count': [],
                   'sentences': [],
                   'keywords':[]}

    
    for i in np.sort(clustered_df.cluster_label.unique()):
        print(f'Working on cluster {i}')
        
        
        cluster_i_df = clustered_df.loc[clustered_df['cluster_label'] == i]
        print(f'There are {cluster_i_df.shape} sentences in cluster {i}') # 40 x 6
        cluster_i_df.split_sent.to_list()
    
        my_lst_str = ' '.join(map(str, cluster_i_df.split_sent.to_list()))
    
        #print(my_lst_str)
        
        # using default model
        kw_model = KeyBERT()
        
        # using roberta model
        # kw_model = KeyBERT(model='roberta-large-nli-stsb-mean-tokens')
        
        
        #print(f"The keywords for cluster {i} are: ")
        #keywords = kw_model.extract_keywords(my_lst_str)
        #print(keywords)
        print(f"The keywords for cluster {i} are: ")
        
        # top keywords in bigrams
        keywords = kw_model.extract_keywords(my_lst_str, keyphrase_ngram_range=(ngram_low, ngram_high), stop_words=None)
        
        # max sum similarity
        # kw_phrases = kw_model.extract_keywords(my_lst_str, keyphrase_ngram_range=(1, 2), stop_words='english', 
        #                           use_maxsum=True, nr_candidates=20, top_n=5)
        
        print(keywords)
    
        #item_cl_kw_dict['item'].append(item)
        item_cl_kw_dict['cl_sentence_count'].append(len(cluster_i_df.split_sent.to_list()))
        item_cl_kw_dict['cluster'].append(i)
        item_cl_kw_dict['sentences'].append(my_lst_str)
        item_cl_kw_dict['keywords'].append(keywords)
        
    
    cluster_summary_df = pd.DataFrame(item_cl_kw_dict)
    
    return cluster_summary_df



civ_summary_df = cluster_keybert(civ_size_n_df, 1, 3)

civ_summary_df.shape
civ_summary_df.columns

civ_summary_df.to_csv("civ_summary_df_trigrams_20210916.csv")



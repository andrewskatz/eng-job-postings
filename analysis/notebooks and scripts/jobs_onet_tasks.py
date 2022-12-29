# -*- coding: utf-8 -*-
"""
Created on Fri Dec 23 14:51:07 2022

@author: akatz4
"""

topic = "jobs"
# item options are civil, mechanical, environmental, biomedical, electrical, chemical
# item = "civil"
item = "mechanical"

from personal_utilities import label_cluster as lc



import pickle
import numpy as np
import os
import pandas as pd
import re
# import seaborn as sns
# import umap 
# import umap.plot
# import hdbscan
import spacy
from spacy.lang.en import English

# from sklearn.manifold import MDS, TSNE




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
    





"""

Import data and preprocess

"""

# =============================================================================
# job postings data
# =============================================================================

os.getcwd()

# for jee df with abstracts and full article information
proj_path = "G:/My Drive/AK Faculty/Research/Projects/project political economy of engineering education/project engineering jobs and skills/data/discipline jobs postings"

os.chdir(proj_path)
os.listdir()


if item == "civil":
    jobs_df = pd.read_csv("no_dupes_civ_eng_20210409.csv")
elif item == "mechanical":
    jobs_df = pd.read_csv("no_dupes_mechanical_us_20210411.csv")
elif item == "electrical":
    jobs_df = pd.read_csv("no_dupes_electrical_engineering_US.csv")
elif item == "biomedical":
    jobs_df = pd.read_csv("no_dupes_biomedical_us_20210410.csv")
elif item == "chemical":
    jobs_df = pd.read_csv("no_dupes_chem_eng_20210409.csv")
elif item == "environmental":
    jobs_df = pd.read_csv("no_dupes_environmental_us_20210410.csv")




# =============================================================================
# onet tasks data
# =============================================================================

label_ver = "tasks"

onet_path = "G:/My Drive/AK Faculty/Research/Projects/project political economy of engineering education/project engineering jobs and skills/data/onet"
os.chdir(onet_path)

if label_ver == "tasks":
    labels_df = pd.read_csv("Task Statements.csv")
    print(labels_df.columns)
    print(labels_df.shape) #19259 x 8
    labels_df = labels_df.drop_duplicates(subset=['Task']) # 17976 x 8


class_labels = list(labels_df.Task.dropna().unique())
print(len(class_labels))
print(class_labels)

labels_df.columns
labels_df['new_label_id'] = labels_df.index

# =============================================================================
# Start processing the jobs data
# =============================================================================

# from 4758 postings there were 115936 sentences -- too many
# try randomly sampling 1000 jobs for sentence segmentation

jobs_df.columns

jobs_filtered_df, jobs_filtered_list = select_and_filter(jobs_df, 'description')
#4758x9



sample_size = 1000

sampled_jobs_df = jobs_filtered_df.sample(n=sample_size, random_state=888)


# First, parse on sentences in each award

jobs_sentence_df, jobs_sent_list = sentence_segmenter(sampled_jobs_df, 'description')



## add batch id to sentences df

group_size = 100
unlabeled_df = jobs_sentence_df.reset_index(drop=True)

# unlabeled_df.apply(lambda row: math.floor(row.index / group_size), axis = 1)
unlabeled_df['new_index'] = unlabeled_df.index
unlabeled_df['batch_num'] = np.floor(unlabeled_df['new_index'] / group_size)


print(unlabeled_df.columns)





# =============================================================================
# label using top score
# =============================================================================



unlabeled_text_col_name = 'split_sent'
unlabeled_id_col_name = 'new_index'
example_text_col = 'Task'
example_label_col = 'Task'

similarity_type = 'top_score'
sim_n=5



sim_ex_df, bad_clusters = lc.label_the_unlabeled(labels_df, 
                                                 unlabeled_df,
                                                 full_embeddings='',
                                                 need_to_embed=True,
                                                 unlabeled_id_col = unlabeled_id_col_name,
                                                 unlabeled_text_col=unlabeled_text_col_name,
                                                 unlabeled_cluster_label_col='batch_num',
                                                 example_id_col='new_label_id',
                                                 example_text_col=example_text_col,
                                                 example_label_col=example_label_col,
                                                 embedding_model = 'all-mpnet',
                                                 similarity_type=similarity_type,
                                                 sim_n=sim_n)



run_date = "20221223"

save_path = "G:/My Drive/AK Faculty/Research/Projects/project political economy of engineering education/project engineering jobs and skills/analysis/task and job descriptions similarities"
os.chdir(save_path)

sim_ex_df.to_csv(f"{topic}_{item}_{similarity_type}_{run_date}_{label_ver}labels_samp-size{sample_size}.csv", index=False)













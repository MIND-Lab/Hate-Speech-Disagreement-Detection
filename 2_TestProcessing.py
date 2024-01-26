##################################
#  Requirements:
#--------------------------------
# The same code has been executed on all the dataset
# The actual code refers to ConvAbuse. 
# To execute on different data, adjust the data path.
#
# Training data should be in a folder named "Data"
# Data paths can be specified at lines 47, 90
# saving paths at lines 156 and 157 
#--------------------------------
#  What does the code do:
#--------------------------------
# Select all the tokens that satisfy the pos tagging requirements both from the
# training and the test dataset, and compute their embeddings.
# Computes the score for every element in the test dataset.
# Saves initial temporary scores for the token in the test dataset (to
# memorize the list of valid tokens) and the distances between each pair of
# tokens (a csv file per instance). 
#--------------------------------
#  Notes:
#--------------------------------
# We selected 'bert-base-multilingual-cased', but we also provided code to 
# execute with 'bert-base-uncased' at line 39
##################################

import warnings
warnings.filterwarnings("ignore")
from Utils import preprocessing,  pos_methods
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from collections import OrderedDict

#Utils
from transformers import AutoTokenizer, BertModel
#model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states = True)
#tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained('bert-base-multilingual-cased', output_hidden_states = True)
tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")


# Train preprocessing
train_df = pd.read_json("./Data/ConvAbuse_train.json", orient='index')
train_df = preprocessing.get_dataset_labels(train_df)


context_embeddings = []
context_tokens = []
sentence_number = []
sentence_score = []

number=1
for sentence in tqdm(list(train_df['original_text'])):
  tokenized_text, list_token_embeddings = pos_methods.text_to_emb(sentence, tokenizer, model)

  acceptable_pos = pos_methods.clear_tokens_pos(sentence, tokenized_text)
  
  # make ordered dictionary to keep track of the position of each word
  tokens = OrderedDict()

  # loop over tokens in sensitive sentence
  for token in tokenized_text:
    if pos_methods.acceptable_token(token) and token in acceptable_pos:
      # keep track of position of word and whether it occurs multiple times
      if token in tokens:
        tokens[token] += 1
      else:
        tokens[token] = 1

      # compute the position of the current token
      token_indices = [i for i, t in enumerate(tokenized_text) if t == token]
      current_index = token_indices[tokens[token]-1]

      # get the corresponding embedding
      token_vec = list_token_embeddings[current_index]

      # save values
      context_tokens.append(token)
      context_embeddings.append(token_vec)
      sentence_number.append(number)
      sentence_score.append(train_df.loc[number,'disagreement'])
  number =number +1

print('len train:' + str(len(context_tokens)))

#Load test, compute scores on test
test_df = pd.read_json("./Data/ConvAbuse_test.json", orient='index')
test_df = preprocessing.get_dataset_labels(test_df)

scores_df_test = pd.DataFrame(columns=['token','#sample','sentence_label','score'])
context_tokens_test = []
sentence_number_test = []
sentence_score_test = []

number=1
for sentence in tqdm(test_df['original_text'].values):
  tokenized_text, list_token_embeddings = pos_methods.text_to_emb(sentence, tokenizer, model)

  acceptable_pos = pos_methods.clear_tokens_pos(sentence, tokenized_text)
  
  # make ordered dictionary to keep track of the position of each word
  tokens = OrderedDict()

  # loop over tokens in sensitive sentence
  for token in tokenized_text:
    if pos_methods.acceptable_token(token) and token in acceptable_pos:
      # keep track of position of word and whether it occurs multiple times
      if token in tokens:
        tokens[token] += 1
      else:
        tokens[token] = 1

      # compute the position of the current token
      token_indices = [i for i, t in enumerate(tokenized_text) if t == token]
      current_index = token_indices[tokens[token]-1]

      # get the corresponding embedding
      token_vec = list_token_embeddings[current_index]

      # save values
      context_tokens_test.append(token)
      sentence_number_test.append(number)
      sentence_score_test.append(test_df.loc[number,'disagreement'])
  number =number +1


scores_df_test['token']=context_tokens_test
scores_df_test['#sample']=sentence_number_test
scores_df_test['sentence_label']=sentence_score_test

print('len text:' + str(len(context_tokens_test)))

threshold=0.7
for index, row in tqdm(test_df.iterrows()):

  _, distances_df, new_words = pos_methods.find_similar_words(test_df.loc[index,'original_text'], pos_methods.selected_tokens_indexes(test_df.loc[index,'original_text'], tokenizer, model)[0], tokenizer, context_tokens,context_embeddings, sentence_score, sentence_number, model)
  #print(similar_words)
  closer_terms = []
  for word in new_words:
    selected_neighbours = distances_df.loc[(distances_df['sim_token']==word) & (distances_df['distance']>=threshold)& (distances_df['distance']!=1)].sort_values(by=['distance'], ascending=False)#[1:] #escludo il primo che è il match perfetto
    if len(selected_neighbours.loc[selected_neighbours['token_label']==1,'distance'])>0:
      pos_score = np.sum(selected_neighbours.loc[selected_neighbours['token_label']==1,'distance'])/np.sum([item for sublist in selected_neighbours[['distance']].values for item in sublist] )
    else:
      pos_score = 0
    if len(selected_neighbours.loc[selected_neighbours['token_label']==0,'distance'])>0:
      neg_score = np.sum(selected_neighbours.loc[selected_neighbours['token_label']==0,'distance'])/np.sum([item for sublist in selected_neighbours[['distance']].values for item in sublist] )
    else:
      neg_score = 0

    stimated_coordinate = pos_score - neg_score

    scores_df_test.loc[(scores_df_test['token']==word)&(scores_df_test['#sample']==index), 'score'] = stimated_coordinate
  scores_df_test.to_csv('./results/scores_df_test_ConvAbuse.csv', sep='\t', index=False)
  distances_df.to_csv('./results/distances_ConvAbuse/dist_'+str(index)+'.csv', sep='\t', index=False) 

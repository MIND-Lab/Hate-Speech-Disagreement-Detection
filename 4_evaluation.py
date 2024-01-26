##################################
#  Requirements:
#--------------------------------
# The same code has been executed on all the dataset
# The actual code refers to ConvAbuse. 
# To execute on different data, adjust the data path.
#
# Training data should be in a folder named "Data"
# data paths can be specified at lines 35 and 55, 157 and 172
#--------------------------------
#  What does the code do:
#--------------------------------
# Estimates the score on the test set using the best thresholds (neighborhood 
# and predictions) estimated on the validation dataset.
# Make predictions with the four proposed approaches: Sum, Mean, Median, and Min
# Save predictions and results.
# Saving paths at lines 216 and 242
##################################


import pandas as pd
import numpy as np
import warnings
from tqdm import tqdm
warnings.filterwarnings("ignore")
from sklearn.metrics import classification_report
from Utils import preprocessing

def get_scores(sentence_num, scores_df):
    return list(scores_df.loc[scores_df['#sample']==sentence_num, 'score'].values), list(scores_df.loc[scores_df['#sample']==sentence_num, 'token'].values)

dev_df = pd.read_json("./Data/ConvAbuse_dev.json", orient='index')
dev_df = preprocessing.get_dataset_labels(dev_df)

scores_df_dev= pd.read_csv('./results/scores_df_dev_ConvAbuse.csv', sep='\t')
#scores_df_dev = scores_df_dev[(scores_df_dev.token != 'prev') & (scores_df_dev.token != 'agent')]

somma_threshold_neghborhood = 0
media_threshold_neghborhood = 0
mediana_threshold_neghborhood = 0
min_threshold_neghborhood = 0
somma_global_best_th = 0
media_global_best_th = 0
mediana_global_best_th = 0
min_global_best_th = 0
somma_global_best_f1 = 0
media_global_best_f1 = 0
mediana_global_best_f1 = 0
min_global_best_f1 = 0

for threshold in [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]:
  scores_df_dev['score'] = 0
  for index, row in dev_df.iterrows():
    
    distances_df = pd.read_csv('./results/distances_ConvAbuse/dist_dev_'+str(index)+'.csv', sep='\t')
    #distances_df = distances_df[(distances_df.sim_token != 'prev') & (distances_df.sim_token != 'agent')]
    closer_terms = []

    new_words = list(distances_df.loc[(distances_df['sim_token']=='Please')&(distances_df['sim_token_sentence_number']==1), 'new_token'])


    for i in range(0, len(new_words)):

      word = new_words[i]
      word_distances = distances_df.iloc[[len(new_words)*a+i for a in range(0, round(distances_df.shape[0]/len(new_words)) )]]
      
      selected_neighbours = word_distances.loc[(word_distances['distance']>=threshold)&(word_distances['distance']!=1)]
      
      if len(selected_neighbours.loc[selected_neighbours['token_label']==1,'distance'])>0:
        pos_score = np.sum(selected_neighbours.loc[selected_neighbours['token_label']==1,'distance'])/np.sum([item for sublist in selected_neighbours[['distance']].values for item in sublist] )
      else:
        pos_score = 0
      if len(selected_neighbours.loc[selected_neighbours['token_label']==0,'distance'])>0:
        neg_score = np.sum(selected_neighbours.loc[selected_neighbours['token_label']==0,'distance'])/np.sum([item for sublist in selected_neighbours[['distance']].values for item in sublist] )
      else:
        neg_score = 0

      stimated_coordinate = pos_score - neg_score
      scores_df_dev.loc[scores_df_dev.loc[scores_df_dev['#sample']==index, 'score'].index[i], 'score'] = stimated_coordinate

  pred_somma = []
  pred_tutti_verdi = []
  pred_media = []
  pred_mediana = []

  for index, _ in dev_df.iterrows():
    colors_agreement = get_scores(index, scores_df_dev)[0]

    if colors_agreement:
      pred_somma.append(sum(colors_agreement))
      pred_media.append(np.mean(colors_agreement))
      pred_mediana.append(np.median(colors_agreement))
      pred_tutti_verdi.append(min(colors_agreement))
    else:
      pred_somma.append(0)
      pred_media.append(0)
      pred_mediana.append(0)
      pred_tutti_verdi.append(0)

  best_t_somma = 0
  best_f1 = 0
  for t in np.arange(round(min(pred_somma)), round(max(pred_somma)), 0.1):
    report = classification_report(dev_df['disagreement'], [int(i>=t) for i in pred_somma], output_dict=True)
    if report['macro avg']['f1-score'] > best_f1:
      best_f1 = report['macro avg']['f1-score']
      best_t_somma = t

    if best_f1 > somma_global_best_f1:
      somma_global_best_f1 = best_f1
      somma_threshold_neghborhood = threshold
      somma_global_best_th = best_t_somma

  best_t_media = 0
  best_f1_media = 0
  pred = pred_media
  for t in np.arange(round(min(pred)), round(max(pred)), 0.1):
    t = round(t,1)
    report = classification_report(dev_df['disagreement'], [int(i>=t) for i in pred], output_dict=True)
    if report['macro avg']['f1-score'] > best_f1_media:
      best_f1_media = report['macro avg']['f1-score']
      best_t_media = t
    if best_f1_media > media_global_best_f1:
      media_global_best_f1 = best_f1_media
      media_threshold_neghborhood = threshold
      media_global_best_th = best_t_media
      
  best_t_mediana = 0
  best_f1_mediana = 0
  pred = pred_mediana
  for t in np.arange(round(min(pred)), round(max(pred)), 0.1):
    t = round(t,1)
    report = classification_report(dev_df['disagreement'], [int(i>=t) for i in pred], output_dict=True)
    if report['macro avg']['f1-score'] > best_f1_mediana:
      best_f1_mediana = report['macro avg']['f1-score']
      best_t_mediana = t
    if best_f1_mediana > mediana_global_best_f1:
      mediana_global_best_f1 = best_f1_mediana
      mediana_threshold_neghborhood = threshold
      mediana_global_best_th = best_t_mediana
      
  best_t_verdi = 0
  best_f1_verdi = 0
  pred = pred_tutti_verdi
  for t in np.arange(round(min(pred)), round(max(pred)), 0.1):
    t = round(t,1)
    report = classification_report(dev_df['disagreement'], [int(i>=t) for i in pred], output_dict=True)
    if report['macro avg']['f1-score'] > best_f1_verdi:
      best_f1_verdi = report['macro avg']['f1-score']
      best_t_verdi = t
    if best_f1_verdi > min_global_best_f1:
      min_global_best_f1 = best_f1_verdi
      min_threshold_neghborhood = threshold
      min_global_best_th = best_t_verdi
      


test_df = pd.read_json("./Data/ConvAbuse_test.json", orient='index')
test_df = preprocessing.get_dataset_labels(test_df)

scores_df_test= pd.read_csv('./results/scores_df_test_ConvAbuse.csv', sep='\t')

scores_df_test['score'] = None

prediction_df = pd.DataFrame(columns=['original_text', 'disagreement', 'somma', 'media', 'mediana', 'min', 'somma_t', 'media_t', 'mediana_t', 'min_t'])
prediction_df['original_text'] = test_df['original_text']
prediction_df['disagreement'] = test_df['disagreement']

for threshold in [somma_threshold_neghborhood, media_threshold_neghborhood, mediana_threshold_neghborhood, min_threshold_neghborhood]:

  for index, row in tqdm(test_df.iterrows()):
    
    distances_df = pd.read_csv('./results/distances_ConvAbuse/dist_'+str(index)+'.csv', sep='\t')
    #distances_df = distances_df[(distances_df.sim_token != 'prev') & (distances_df.sim_token != 'agent')]
    closer_terms = []

    new_words = list(distances_df.loc[(distances_df['sim_token']=='Please')&(distances_df['sim_token_sentence_number']==1), 'new_token'])

    for i in range(0, len(new_words)):

      word = new_words[i]
      word_distances = distances_df.iloc[[len(new_words)*a+i for a in range(0, round(distances_df.shape[0]/len(new_words)) )]]

      selected_neighbours = word_distances.loc[(word_distances['distance']>=threshold)&(word_distances['distance']!=1)]
      
      if len(selected_neighbours.loc[selected_neighbours['token_label']==1,'distance'])>0:
        pos_score = np.sum(selected_neighbours.loc[selected_neighbours['token_label']==1,'distance'])/np.sum([item for sublist in selected_neighbours[['distance']].values for item in sublist] )
      else:
        pos_score = 0
      if len(selected_neighbours.loc[selected_neighbours['token_label']==0,'distance'])>0:
        neg_score = np.sum(selected_neighbours.loc[selected_neighbours['token_label']==0,'distance'])/np.sum([item for sublist in selected_neighbours[['distance']].values for item in sublist] )
      else:
        neg_score = 0

      stimated_coordinate = pos_score - neg_score
      scores_df_test.loc[scores_df_test.loc[scores_df_test['#sample']==index, 'score'].index[i], 'score'] = stimated_coordinate

  pred_somma = []
  pred_media= []
  pred_mediana = []
  pred_tutti_verdi = []

  for index, _ in test_df.iterrows():
    colors_agreement = get_scores(index, scores_df_test)[0]

    if colors_agreement:
      pred_somma.append(sum(colors_agreement))
      pred_media.append(np.mean(colors_agreement))
      pred_mediana.append(np.median(colors_agreement))
      pred_tutti_verdi.append(min(colors_agreement))
    else:
      pred_somma.append(0)
      pred_media.append(0)
      pred_mediana.append(0)
      pred_tutti_verdi.append(0)
  
  with open('./ConvAbuse/results_ConvAbuse.txt', 'a') as f:
    f.write('SUM \n')
    f.write(classification_report(test_df['disagreement'], [int(i>=somma_global_best_th) for i in pred_somma] ))

    f.write('MEAN \n')
    f.write(classification_report(test_df['disagreement'], [int(i>=media_global_best_th) for i in pred_media] ))
  
    f.write('MEDIAN \n')
    f.write(classification_report(test_df['disagreement'], [int(i>=mediana_global_best_th) for i in pred_mediana] ))
  
    f.write('MIN \n')
    f.write(classification_report(test_df['disagreement'], [int(i>=min_global_best_th) for i in pred_tutti_verdi] ))

  if threshold == somma_threshold_neghborhood:
    prediction_df['somma'] = pred_somma
    prediction_df['somma_t'] = [int(i>=somma_global_best_th) for i in pred_somma]
  if threshold == media_threshold_neghborhood:
    prediction_df['media'] = pred_media
    prediction_df['media_t'] = [int(i>=media_global_best_th) for i in pred_media] 
  if threshold == mediana_threshold_neghborhood:
    prediction_df['mediana'] = pred_mediana
    prediction_df['mediana_t'] = [int(i>=mediana_global_best_th) for i in pred_mediana]
  if threshold == min_threshold_neghborhood:
    prediction_df['min'] = pred_tutti_verdi
    prediction_df['min_t'] = [int(i>=min_global_best_th) for i in pred_tutti_verdi]

prediction_df.to_csv('predictions_df_ConvAbuse.csv', index=False, sep='\t')

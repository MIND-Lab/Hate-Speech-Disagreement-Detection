import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import classification_report
from Utils import preprocessing

def get_scores(sentence_num, scores_df):
    return list(scores_df.loc[scores_df['#sample']==sentence_num, 'score'].values), list(scores_df.loc[scores_df['#sample']==sentence_num, 'token'].values)

dev_df = pd.read_json("Data/ConvAbuse_dev.json", orient='index')
dev_df = preprocessing.get_dataset_labels(dev_df)

scores_df_dev= pd.read_csv('results/scores_df_dev_ConvAbuse.csv', sep='\t')

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

for threshold in [0.5, 0.55, 0.6, 0.65, 0.7, 0,75, 0.8, 0.85, 0.9, 0.95]:
  #scores_df_dev['score'] = 0
  for index, row in dev_df.iterrows():
    
    distances_df = pd.read_csv('./results/distances_ConvAbuse/dist_dev_'+str(index)+'.csv', sep='\t')
    distances_df = distances_df[(distances_df.sim_token != 'prev') & (distances_df.sim_token != 'agent')]
    distances_df = distances_df[(distances_df.new_token != 'prev') & (distances_df.new_token != 'agent')]
    closer_terms = []

    new_words = list(distances_df.loc[(distances_df['sim_token']=='Please')&(distances_df['sim_token_sentence_number']==1), 'new_token'])


    for i in range(0, len(new_words)):

      word = new_words[i]
      #sistemato per distinguere le diverse occorrenze del termine
      word_distances = distances_df.iloc[[len(new_words)*a+i for a in range(0, round(distances_df.shape[0]/len(new_words)) )]]

      # rimesso il più vicino perchè il confronto viene fatto con il train, quindi non avrò mai il match perfetto (poi tolto ancora perchè funziona meglio)
      #selected_neighbours = word_distances.loc[word_distances['distance']>=threshold].sort_values(by=['distance'], ascending=False)[1:]
      
      #nuovo: 
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

      #sistemato per avere uno score diverso se un termine compare più volte nella stessa frase
      #nuovo: 
      scores_df_dev.loc[scores_df_dev.loc[scores_df_dev['#sample']==index, 'score'].index[i], 'score'] = stimated_coordinate
      #scores_df_dev.loc[(scores_df_dev['token']==word)&(scores_df_dev['#sample']==index), 'score'] = stimated_coordinate

  pred_somma = []
  pred_tutti_verdi = []
  pred_media = []
  pred_mediana = []

  for index, _ in dev_df.iterrows():
    colors_agreement = get_scores(index, scores_df_dev)[0]

    #tolgo gli zero:
    colors_agreement = [i for i in colors_agreement if i != 0]

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
      
      print('SOMMA \n')
      print('THRESHOLD: '+ str(best_t_somma) + '\n')
      print(classification_report(dev_df['disagreement'], [int(i>=best_t_somma) for i in pred_somma] ))


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
      
      print('MEDIA \n')
      print('THRESHOLD: '+ str(best_t_media) + '\n')
      print(classification_report(dev_df['disagreement'], [int(i>=best_t_media) for i in pred] ))



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
      
      print('MEDIANA \n')
      print('THRESHOLD: '+ str(best_t_mediana) + '\n')
      print(classification_report(dev_df['disagreement'], [int(i>=best_t_mediana) for i in pred] ))

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
      
      print('MEDIA \n')
      print('THRESHOLD: '+ str(best_t_verdi) + '\n')
      print(classification_report(dev_df['disagreement'], [int(i>=best_t_verdi) for i in pred] ))

print(somma_threshold_neghborhood)
print(media_threshold_neghborhood)
print(mediana_threshold_neghborhood)
print(min_threshold_neghborhood)
print(somma_global_best_th)
print(media_global_best_th)
print(mediana_global_best_th)
print(min_global_best_th)
print(somma_global_best_f1)
print(media_global_best_f1)
print(mediana_global_best_f1)
print(min_global_best_f1)
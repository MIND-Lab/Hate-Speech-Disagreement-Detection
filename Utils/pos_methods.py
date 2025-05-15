import spacy
import string
from scipy.spatial.distance import cosine
from . import BERT_Embeddings
import torch
import pandas as pd

nlp = spacy.load("en_core_web_sm")
pos_to_keep = ['ADJ','ADV','INTJ','NOUN','PRON','PROPN','VERB', 'X']#,'DET'
punctuations = list(string.punctuation)

def clear_pos(sentence, nlp=nlp, pos_to_keep= pos_to_keep):
    doc = nlp(sentence)
    keep = []
    remove=[]
    for token in doc:
        if token.pos_ in pos_to_keep:
            keep.append(str(token))
        else:
            remove.append(str(token))
    return keep, remove

def clear_tokens_pos(sentence, tokens):
    """
    takes the sentence and the list of tokens.
    compute the list of tokens to remove according to their pos tagging.
    Return the list of tokens to keep.
    """
    tokens1 = tokens.copy()

    to_remove = clear_pos(sentence)[1]
    # if perfet match remove, otherwise store
    for el in to_remove:
        if el in tokens1:
            tokens1.remove(el)
        elif el == "n't":
          if "'t" in tokens1:
            tokens1.remove("'t")
        elif el == "n’t":
          if "’t" in tokens1:
            tokens1.remove("’t")
    while "'" in tokens1:
      tokens1.remove("'")
    while "’" in tokens1:
      tokens1.remove("’")
    while "’re" in tokens1:
      tokens1.remove("’re")
    while "’s" in tokens1:
      tokens1.remove("’s")
    while "’m" in tokens1:
      tokens1.remove("’m")

    tokens1 = [t for t in tokens1 if not(t.isnumeric())]
    return tokens1

def acceptable_token(token):
  if token not in punctuations and token not in ['user', 'url', 'RT']:
    return True
  return False

def acceptable_token_ConvAbuse(token):
  if token not in punctuations and token not in ['user', 'url', 'RT', "prev_agent", "prev_user","agent"]:
    return True
  return False

def aggregate_subwords(encoding, list_token_embeddings, text):
    recomposed_tokens = []  # List to store the recomposed tokens
    recomposed_emb = []  # List to store the recomposed embeddings
    hashtag = False  # Flag to indicate if a hashtag is encountered
    hashtag_emb = False  # Flag to indicate if a hashtag is part of the mean calculation

    for i in sorted(list(set(encoding.word_ids())), key=lambda x: (x is None, x)):
      #index_of_token = encoding.word_ids()[i]
      if i != None:
        #if the embedding is related to a single token
        if encoding.word_ids().count(i) ==1:
          recomposed_emb.append(list_token_embeddings[encoding.word_ids().index(i)])
        #if the embed is given by the mean of multiple tokens
        elif encoding.word_ids().count(i) >1:
          #retrive the first one
          emb = list_token_embeddings[encoding.word_ids().index(i)]
          # count the number of tokens to mean
          num = encoding.word_ids().count(i)
          # if I have to iclude an hashtag inside a mean
          if hashtag_emb:
            #remove last element (the hashag) and include it in the mean
            emb = emb + recomposed_emb.pop()
            num = encoding.word_ids().count(i)+1
            hashtag_emb = False
          for a in range(1, encoding.word_ids().count(i)):
            emb = emb + list_token_embeddings[encoding.word_ids().index(i)+a]
          emb = emb/num
          recomposed_emb.append(emb)

        start, end = encoding.word_to_chars(i)
        #print(text[start:end])
        if hashtag:
          recomposed_tokens.append('#'+text[start:end])
          hashtag=False
        elif text[start:end] == '#':
          hashtag=True
          hashtag_emb = True
          hash_emb = list_token_embeddings[encoding.word_ids().index(i)]

        else:
          #print(text[start:end])
          recomposed_tokens.append(text[start:end])
    
    while "'" in recomposed_tokens:
      pos = recomposed_tokens.index("'")
      new = ''
      new_emb = torch.tensor([0]*768)
      if pos+2 < len(recomposed_tokens):
        for el in range(pos,pos+2):
          new_emb = new_emb + recomposed_emb[el]
          new=new+recomposed_tokens[el]
      else: #ends with '
        new_emb = new_emb + recomposed_emb[pos]
        new=new+recomposed_tokens[pos]
        break

      recomposed_tokens  = recomposed_tokens[:pos] + [new] + recomposed_tokens[pos+2:]
      recomposed_emb  = recomposed_emb[:pos] + [new_emb/3] + recomposed_emb[pos+2:]

    while "’" in recomposed_tokens:
      pos = recomposed_tokens.index("’")
      new = ''
      new_emb = torch.tensor([0]*768)
      if pos+2 < len(recomposed_tokens):
        for el in range(pos,pos+2):
          new_emb = new_emb + recomposed_emb[el]
          new=new+recomposed_tokens[el]
      else: #ends with ’
        new_emb = new_emb + recomposed_emb[pos]
        new=new+recomposed_tokens[pos]
        break

      recomposed_tokens  = recomposed_tokens[:pos] + [new] + recomposed_tokens[pos+2:]
      recomposed_emb  = recomposed_emb[:pos] + [new_emb/3] + recomposed_emb[pos+2:]

    if "*" in recomposed_tokens:
      pos = recomposed_tokens.index("*")
      new = ''
      new_emb = torch.tensor([0]*768)
      consec = 1
      if pos+consec < len(recomposed_tokens):
          next = recomposed_tokens[pos+consec]
      else: #ends with *
        next = ''
        consec = consec-1
      while (next =='*'):
        consec = consec+1
        if pos+consec < len(recomposed_tokens):
          next = recomposed_tokens[pos+consec]
        else: #ends with *
          next = ''
          consec = consec-1
      for el in range(pos-1,pos+consec+1):
        new_emb = new_emb + recomposed_emb[el]
        new=new+recomposed_tokens[el]

      recomposed_tokens  = recomposed_tokens[:pos-1] + [new] + recomposed_tokens[pos+consec+1:]
      recomposed_emb  = recomposed_emb[:pos-1] + [new_emb/3] + recomposed_emb[pos+consec+1:]
    return recomposed_tokens, recomposed_emb

def text_to_emb(text, tokenizer, model):
  tokenized_text, tokens_tensor, segments_tensors, encoding = BERT_Embeddings.bert_text_preparation(text, tokenizer)
  list_token_embeddings = BERT_Embeddings.get_bert_embeddings(tokens_tensor, segments_tensors, model)
  recomposed_tokens, recomposed_emb = aggregate_subwords(encoding, list_token_embeddings, text)
  return recomposed_tokens, recomposed_emb

def find_similar_words(new_sent, new_token_indexes, tokenizer, context_tokens, context_embeddings, sentence_score, sentence_number, model):
    """
    Find similar words to the given new words in a context.

    Args:
        new_sent (str): The input sentence containing the new words.
        new_token_indexes (list): List of indexes of the new words in the sentence.

    Returns:
        tuple: A tuple containing:
            - similar_words (list): List of similar words for each new word.
            - distances_df (DataFrame): DataFrame containing the token, new_token, and distance.
            - new_words (list): List of the new words extracted from the sentence.
    """

    # embeddings for the NEW word 'record'
    list_of_distances = []
    list_of_new_embs = []
    new_words = []

    tokenized_text, new_emb = text_to_emb(new_sent, tokenizer, model)

    for new_token_index in new_token_indexes:
        #new_emb = get_bert_embeddings(tokens_tensor, segments_tensors, model)[new_token_index]
        list_of_new_embs.append(new_emb[new_token_index])
        new_words.append(tokenized_text[new_token_index])

    #print(new_words)
    for sentence_1, embed1, score_1, number_1 in zip(context_tokens, context_embeddings, sentence_score, sentence_number):
        for i in range(0, len(new_token_indexes)):
            cos_dist = 1 - cosine(embed1, list_of_new_embs[i])
            list_of_distances.append([sentence_1, new_words[i], cos_dist, score_1, number_1])

    distances_df = pd.DataFrame(list_of_distances, columns=['sim_token', 'new_token', 'distance', 'token_label', 'sim_token_sentence_number'])

    similar_words = []

    for i in range(0, len(new_token_indexes)):
      if distances_df.loc[distances_df.new_token == new_words[i], 'distance'].idxmax():
        similar_words.append([distances_df.loc[distances_df.loc[distances_df.new_token == new_words[i], 'distance'].idxmax(), 'sim_token']])
      else:
        similar_words.append([])

    return similar_words, distances_df, new_words

def selected_tokens_indexes(sentence, tokenizer, model):
  #also remove elements according to pos tags
  tokenized_text, _ = text_to_emb(sentence, tokenizer, model)
  #tokenized_text_2 = clear_tokens_pos(sentence, tokenized_text) #list of tokens to keep according to their pos tagging

  i=0
  tokens_indexes=[]
  for token in tokenized_text:
    if acceptable_token(token) and token in clear_tokens_pos(sentence, tokenized_text):
      tokens_indexes.append(i)
    i=i+1
  return tokens_indexes, [tokenized_text[a] for a in tokens_indexes]

from bs4 import BeautifulSoup

def clear_html_sintax(text):
  soup = BeautifulSoup(text)
  # get text
  return soup.get_text()
# Load dependencies

import pandas as pd
import numpy as np
import torch

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples

import matplotlib.pyplot as plt
from tqdm import tqdm

# Load Data

!wget "https://github.com/lhaggerty18/NLP_Exercise3/raw/refs/heads/main/still_small.csv"
!wget "https://github.com/lhaggerty18/NLP_Exercise3/raw/refs/heads/main/yet_small.csv"

# Format data

df_still = pd.read_csv("still_small.csv", header=None)
df_yet = pd.read_csv("yet_small.csv", header=None)

headers =  ["year", "source_type", "source", "text"]
df_still.columns = headers
df_yet.columns = headers

df_yet.head()


# get BERT embeddings model

from transformers import BertModel, BertTokenizer

model = BertModel.from_pretrained('bert-base-uncased',
                                  output_hidden_states = True,
                                  )


model.eval()

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


# function to process text

def bert_text_preparation(text, tokenizer):
  """
  Preprocesses text input in a way that BERT can interpret.
  """
  marked_text = "[CLS] " + text + " [SEP]"
  tokenized_text = tokenizer.tokenize(marked_text)
  indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
  segments_ids = [1]*len(indexed_tokens)

  # convert inputs to tensors
  tokens_tensor = torch.tensor([indexed_tokens])
  segments_tensor = torch.tensor([segments_ids])

  return tokenized_text, tokens_tensor, segments_tensor

# Decades

decade_vec = [
    range(1820,1830), 
    range(1830,1840), 
    range(1840,1850), 
    range(1850,1860), 
    range(1860,1870), 
    range(1870,1880), 
    range(1880,1890), 
    range(1890,1900), 
    range(1900,1910), 
    range(1910,1920), 
    range(1920,1930), 
    range(1930,1940), 
    range(1940,1950), 
    range(1950,1960), 
    range(1960,1970), 
    range(1970,1980), 
    range(1980,1990), 
    range(1990,2000), 
    range(2000,2010), 
    range(2010,2020)
    ]

vec_all = []

for decade in tqdm(decade_vec):
  
  # Filter data to current decade
  
  df_still_fil =  df_still[df_still['year'].isin(decade)]
  df_yet_fil =  df_yet[df_yet['year'].isin(decade)]

  vec_decade = []

  # still 

  for text in df_still_fil['text']:

    # prep text for BERT
    
    tokenized_text, tokens_tensor, segments_tensor = bert_text_preparation(text, tokenizer)
    token_index = tokenized_text.index("still") # get index of target word
    
    # Get embeddings

    with torch.no_grad():

      outputs = model(tokens_tensor, segments_tensor)
      hidden_states = outputs[2]

    token_embeddings = torch.stack(hidden_states, dim=0)
    token_embeddings = torch.squeeze(token_embeddings, dim=1) # remove batch dimension

    token_embeddings = token_embeddings.permute(1,0,2) # [# tokens, # layers, # features]

    sum_vec = torch.sum(token_embeddings[token_index][-4:], dim=0) # get embedding of target word, sum of final 4 layers

    vec_decade.append(sum_vec)



  # yet

  for text in df_yet_fil['text']:

    tokenized_text, tokens_tensor, segments_tensor = bert_text_preparation(text, tokenizer)
    token_index = tokenized_text.index("yet") # get index of target word

    with torch.no_grad():

      outputs = model(tokens_tensor, segments_tensor)
      hidden_states = outputs[2]

    token_embeddings = torch.stack(hidden_states, dim=0)
    token_embeddings = torch.squeeze(token_embeddings, dim=1) # remove batch dimension

    token_embeddings = token_embeddings.permute(1,0,2) # [# tokens, # layers, # features]

    sum_vec = torch.sum(token_embeddings[token_index][-4:], dim=0) # get embedding of target word, sum of final 4 layers

    vec_decade.append(sum_vec)
  vec_all.append(vec_decade)
  
print()


import random


yet_purity_decade = [] 

for i in tqdm(range(len(vec_all))):

  df = pd.DataFrame(vec_all[i])
  scaler = StandardScaler()
  df_scaled = scaler.fit_transform(df) # scale data for PCA

  pca=PCA(n_components=2) # PCA dimensions

  X=pca.fit_transform(df_scaled)

  X_scaled = scaler.fit_transform(X)

  range_n_clusters = list(range(2, 8))
  silhouette_avgs = []
  
  # KMeans, find optimal k

  for n_clusters in range_n_clusters:
      kmeans = KMeans(n_clusters=n_clusters, n_init='auto', random_state=14)
      cluster_labels = kmeans.fit_predict(X_scaled)
      sil_avg = silhouette_score(X_scaled, cluster_labels)
      silhouette_avgs.append(sil_avg)
      #print(f"For n_clusters = {n_clusters}, average silhouette_score = {sil_avg:.3f}")

  k_opt = silhouette_avgs.index(np.max(silhouette_avgs)) + 2
  print(k_opt)

  kmeans = KMeans(n_clusters=k_opt, n_init='auto', random_state=14)
  y_pred = kmeans.fit_predict(X_scaled)
  centroids = kmeans.cluster_centers_

  chart_title = 1820 + 10*i
  
  # Plot clusters
  
  plt.figure(i)
  plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y_pred)
  plt.title("1820-1829")
  plt.title('Decade: '+str(chart_title)+'-'+str(chart_title + 9))
  plt.show()

  random.seed(14)

  yet_pct_list = []
  
  # Print examples from clusters
  
  # Get "Yet Purity"

  for cluster in range(k_opt):
    ind = [i for i, val in enumerate(y_pred) if val == cluster]
    ind_yet = [i for i, val in enumerate(ind) if val > 999]
    ind_still = [i for i, val in enumerate(ind) if val < 1000]

    c_size = len(ind)
    c_size_yet = len(ind_yet)
    c_yet_pct = c_size_yet/c_size

    print(f"Percentage of 'YET' Cluster {cluster}: {c_yet_pct:.0%}")

    if(c_yet_pct > 0):
      text_example = df_yet.iloc[(random.sample(ind_yet, k = 1)[0] + (1000 * i)), df_yet.columns.get_loc('text')]
      print(f"Example of 'YET' in Cluster {cluster}: {text_example}")

    if(c_yet_pct < 1):
      text_example = df_still.iloc[(random.sample(ind_still, k = 1)[0] + (1000 * i)), df_still.columns.get_loc('text')]
      print(f"Example of 'STILL' in Cluster {cluster}: {text_example}")

    yet_pct_list.append(c_yet_pct)



  yet_purity_list = [abs(x-0.5) for x in yet_pct_list]
  yet_purity = (np.mean(yet_purity_list)/0.5)
  
  print(f"Yet Purity, {chart_title}'s: {yet_purity}")





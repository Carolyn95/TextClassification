import math
import numpy as np
import pandas as pd
from collections import Counter
import contractions
import pdb
import re
from nltk.stem import WordNetLemmatizer
# def createOTE(df):
#   """Extract Opinion Target Expression from token index,
#      storing as a new colomn 'ote' to dataframe.

#      Args:
#       df: original dataframe

#      Return:
#       df: processed dataframe with column 'ote'
#   """
#   df['ote'] = df.apply(lambda row: np.nan
#                        if math.isnan(float(row['ote_start'])) else
#                        (0 if row['ote_start'] == row['ote_stop'] else row[
#                            'text'][int(row['ote_start']):int(row['ote_stop'])]),
#                        axis=1)
#   return df


def replaceNA(df):
  """Replace all NaN with string 'NULL'.
     
     Args:
      df: original dataframe

     Returns:
      df: processed dataframe with all NaN replaced by 'NULL'
  """
  df = df.replace(np.nan, 'NULL', regex=True)
  return df


def splitEntityAspect(df):
  """Split category column into two columns 'entity' and 'aspect' with '#'.
     
     Args: 
      df: dataframe with NaN replaced by 'NULL'
     Returns:
      df: dataframe with entity and aspect
  """
  for i, feature in enumerate(['entity', 'aspect']):
    df[feature] = df.apply(
        lambda row: 'NULL'
        if row['category'] == 'NULL' else row['category'].split('#')[i],
        axis=1)
  return df


def preprocess(df):
  """Preprocess dataframe by replacing NaN and spliting category.
  """
  df = replaceNA(df)
  df = splitEntityAspect(df)
  return df


def explore(*dfs):
  """Explore data. Plot data distribution. Recommend to know target counts also. 
  """
  df_all = pd.concat(dfs)

  all_entity_counts = Counter(list(df_all['entity']))
  all_aspect_counts = Counter(list(df_all['aspect']))
  print(all_entity_counts)
  print(all_aspect_counts)
  table = pd.crosstab(df_all['entity'], df_all['aspect'])
  print(table)


def process(*dfs):
  """
     Expand all contractions and replace slang words with substitutions (eg. I'd -> I would, u -> you common usage)=> 
     remove unusual characters and emoticons =>  
     convert words to lowercase (eg. WWE -> wwe) =>  
     remove numbers =>  
     lemmatize words to their base form (eg. computing, computed -> compute)
  """
  lemmatizer = WordNetLemmatizer()
  for df in dfs:
    for i, row in df.iterrows():
      # pdb.set_trace()
      text = row['text']
      # expand contractions and substitute slangs
      text = contractions.fix(text, slang=True)
      # remove unusual characters
      text = re.sub('<[^>]*>', '', text)
      # remove emoticons
      emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
      # convert all words to lowercase
      text = re.sub('[\W]+', ' ', text.lower()) + " ".join(emoticons).replace(
          '-', '')
      # strip numbers
      text = re.sub(r'\d+', '', text)
      # tokenized by white space
      text_tokenized = text.split()
      # lemmatize
      text_lemmatized = ' '.join(
          [lemmatizer.lemmatize(word, pos='v') for word in text_tokenized])
      row['text'] = text_lemmatized
  return dfs

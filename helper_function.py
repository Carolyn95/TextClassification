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
    print(f'Before dataframe length {len(df)}')
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
    # remove empty sentences after processing (eg. original text only contains digits)
    df = df.loc[df['text'].apply(len) != 0]
    print(f'After dataframe length {len(df)}')
  return dfs


def createMappings(*dfs):
  counts = Counter()
  df_all = pd.concat(dfs)
  for text in df_all['text']:
    text = text.split(' ')
    counts.update(text)
  # sort word counts, descending frequency
  sorted_counts = sorted(counts, key=counts.get, reverse=True)
  # data exploration
  helper_iter = [(0, 10, 1, '*** Top 10 Most Frequent Words ***'),
                 (-1, -11, -1, '*** Bottom 10 Most Frequent Words ***')]
  for it in helper_iter:
    print(it[3])
    for i in range(it[0], it[1], it[2]):
      word = sorted_counts[i]
      line = ['Word:', word, 'Occurences', counts[word]]
      print('{:<0} {:<16} {:<0} {:<0}'.format(*line))
    print('\n')

  # create word to index dictionary, mapping unique word to integer
  word_to_index = {
      word: index for index, word in enumerate(sorted_counts, start=2)
  }
  # add padding token to 1st position of the vocab
  word_to_index['<PAD>'] = 0
  # add OOV (out-of-vocabulary) token to 2nd position of the vocab
  word_to_index['<OOV>'] = 1
  # create index to word dictionary, mapping integer to unique word
  index_to_word = {v: k for k, v in word_to_index.items()}
  return word_to_index, index_to_word


def getEmbeddingMatrix(model_path, word_to_index, vocab_size=None):
  """Get pre-trained GloVe embedding for every word.
     If word not in pre-trained vocabulary, will return a random numeric vector to represent.

     Args:
      model_path: path to pre-trained word2vec model 
      word_to_index: dictionary of {word:index}
      vocab_size: default to length of word_to_index
     Returns:
      embedding_matrix: an embedding matrix of words in target text set
  """
  # load embeddings in a dictionary
  embeddings = {}
  with open(model_path, 'r', encoding='utf-8') as file:
    for line in file:
      values = line.strip().split()
      word = values[0]
      vector = np.asarray(values[1:], dtype='float32')
      embeddings[word] = vector
  # randomly initialize the embedding matrix (vocab_size * embedding_size)
  vocab_size = vocab_size if vocab_size else len(word_to_index)
  embedding_size = len(list(embeddings.values())[0])
  np.random.seed(2021)
  embedding_matrix = np.random.uniform(-1, 1, size=(vocab_size, embedding_size))

  # replace row with corresponding embedding vector if a given word is within the embedding vocab
  num_loaded = 0
  for word, index in word_to_index.items():
    vec = embeddings.get(word)
    if vec is not None and index < vocab_size:
      embedding_matrix[index] = vec
      num_loaded += 1

  # finish and print how many words were loaded
  print('Successfully loaded pretrained embeddings for %f/%f words.' %
        (num_loaded, vocab_size))
  print(
      'Embedding vectors for non-loaded words were randomly initialized between -1 and 1.'
  )
  embedding_matrix = embedding_matrix.astype(np.float32)
  return embedding_matrix


def getMaxLength(*dfs):
  df_all = pd.concat(dfs)
  text_length = max(df_all['text'].apply(lambda x: x.split()).apply(len))
  print(f'Maximum token length is {text_length}')
  return text_length


def padSequence(df, pad_length, pad_direction='left', pad_token='<PAD>'):
  df['tokenized_text'] = df['text'].apply(lambda x: x.split(' '))
  if pad_direction == 'right':
    df['tokenized_text'].apply(
        lambda x: x.extend(pad_token for _ in range(pad_length - len(x))))
  elif pad_direction == 'left':
    df['tokenized_text'] = df['tokenized_text'].apply(
        lambda x: [pad_token for _ in range(pad_length - len(x))] + x)
  else:
    raise ValueError('Please specify a direction for padding.')

  return df


def mapWord2Idx(text, word_to_index):
  """Map from word to index.
  """
  if isinstance(text, str):
    text = text.split(' ')
  mapped_text = [word_to_index[token] for token in text]
  # print(mapped_text)
  return mapped_text


def mapIdx2Word(indexes, index_to_word):
  """Map from index to word.
  """
  mapped_text = [index_to_word[index] for index in indexes]
  # print(mapped_text)
  return mapped_text


def getTagsIOB(text_seq, target_seq, ote_start, ote_stop):
  """Get IOB2 tags for input sequence.

     Args:
     text_seq: a sequence of words
     target_seq: sequence of aspect target words
     ote_start: original starting target character index of the aspect target sequence (before preprocessing)
     ote_stop: original ending character index of the aspect target sequence (before preprocessing)

     Returns:
     tag_seq: a one dimentional numpy array of IOB2 tags for the input
  """

  # helper function to get index which is the closest to ***
  def getClosestSeqIndex(text_seq, ote_start, buffer=0):
    """Find 1st aspect word by accumulatively adding the characters
       in each sequence step and stop at the index where the summation
       matches or exceeds ote_start.
       Slide the target sequence across the text sequence and look for
       the index where they match.

       Args:
        text_seq: a sequence of strings
        ote_start:

       Returns:
        tag_seq:
    """
    sum_ = 0
    for idx in range(len(text_seq)):
      sum_ += len(text_seq[idx])
      if sum_ >= (int(ote_start - buffer)):
        return idx
    # if return never triggers
    raise ValueError(
        "Sum of characters never reaches ote_start, increase buffer.")

  tag_seq = ['O' for word in text_seq]
  tag_seq = np.array(tag_seq)
  target_seq = target_seq.split()
  if target_seq == ['null'] or ote_stop == 0 or ote_start == 'NULL':
    return tag_seq
  matches = []
  target_seq_len = len(target_seq)
  # match a single word
  if target_seq_len == 1:
    target_seq = target_seq[0]  #empty list so dtype is string
    for idx_ in range(len(text_seq)):
      if text_seq[idx_] == target_seq:
        matches.append(idx_)
    if len(matches) == 1:
      tag_seq[matches[0]] = 'B'
    elif len(matches) > 1:
      # use ote_start guess where the index should be
      approximated_index = getClosestSeqIndex(text_seq, ote_start, buffer=5)
      closet_index = min(matches, key=lambda x: abs(x - approximated_index))
      tag_seq[closest_index] = 'B'
    else:
      raise ValueError('Matches are empty for sequence', text_seq,
                       'for target sequence', target_seq)
  # match a sequence
  else:
    for idx_ in range(len(text_seq) - target_seq_len + 1):
      if text_seq[idx_:(idx_ + target_seq_len)] == target_seq:
        matches.append(idx_)
    if len(matches) == 1:
      matched_index = matches[0]
      tag_seq[matched_index:(matched_index + target_seq_len)] = 'I'
      tag_seq[matched_index] = 'B'
    elif len(matches) > 1:
      approximated_index = getClosestSeqIndex(text_seq, ote_start, buffer=5)
      closest_index = min(matches, key=lambda x: abs(x - approximated_index))
      tag_seq[closest_index:(closest_index + target_seq_len)] = 'I'
      tag_seq[closest_index] = 'B'
    else:
      raise ValueError('matches is empty for sequence:', text_seq,
                       'for target sequence:', target_seq)
  return tag_seq


def generateTagsIOB(df, padded_seq):
  """Generate IOB2 tags.
  """
  y_tags = np.empty_like(padded_seq, dtype=object)
  for idx in range(df.shape[0]):
    y_tags[idx] = getTagsIOB(text_seq=padded_seq[idx],
                             target_seq=df['target'].iloc[idx],
                             ote_start=df['ote_start'].iloc[idx],
                             ote_stop=df['ote_stop'].iloc[idx])
  return y_tags

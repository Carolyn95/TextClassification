from parse_data import data_xml_to_df
from helper_function import *
# https://rpmarchildon.com/wp-content/uploads/2018/10/RM-W-NLP-AspectEntity-vF1.html

train_file = 'data/ABSA16_Restaurants_Train_SB1_v2.xml'
test_file = 'data/restaurants_trial_english_sl.xml'

if __name__ == '__main__':
  df_train = data_xml_to_df(train_file)
  df_test = data_xml_to_df(test_file)
  df_test = preprocess(df_test)
  df_train = preprocess(df_train)
  explore(df_train, df_test)
  print(len(df_train), len(df_test))
  df_train, df_test = process(df_train, df_test)
  print(df_train.loc[42, 'text'])
  # Word to index and Index to word transform
  word_to_index, index_to_word = createMappings(df_train, df_test)

  # GloVe 100d model
  embedding_matrix = getEmbeddingMatrix(model_path='GloVe.6B/glove.6B.100d.txt',
                                        word_to_index=word_to_index)
  print('Embedding Matrix Shape:', embedding_matrix.shape)

  # define vocab_size and embedding_size variables for later use
  vocab_size = embedding_matrix.shape[0]
  embedding_size = embedding_matrix.shape[1]
  print('Vocabulary Size:', vocab_size)
  print('Embedding Size:', embedding_size)
  pad_length = getMaxLength(df_train, df_test)
  df_train = padSequence(df_train,
                         pad_length,
                         pad_direction='left',
                         pad_token='<PAD>')
  df_test = padSequence(df_test,
                        pad_length,
                        pad_direction='left',
                        pad_token='<PAD>')

  # indexes = mapWord2Idx(df_test.loc[42, 'tokenized_text'], word_to_index)
  # tokens = mapIdx2Word(indexes, index_to_word)
  # Takes in dataframe, after processing, take the elements out as arrays
  padded_train_text = np.array()
  padded_test_text = 
  print()
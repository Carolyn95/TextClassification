from parse_data import data_xml_to_df
from helper_function import *

train_file = 'data/ABSA16_Restaurants_Train_SB1_v2.xml'
test_file = 'data/restaurants_trial_english_sl.xml'

if __name__ == '__main__':
  df_train = data_xml_to_df(train_file)
  df_test = data_xml_to_df(test_file)
  df_test = preprocess(df_test)
  df_train = preprocess(df_train)
  explore(df_train, df_test)
  print(len(df_train), len(df_test))
  df_new_train, df_new_test = process(df_train, df_test)
  print(df_new_train.loc[42, 'text'])
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
  pdb.set_trace()
  print()
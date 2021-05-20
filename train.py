from parse_data import data_xml_to_df
from helper_function import *

train_file = 'data/ABSA16_Restaurants_Train_SB1_v2.xml'
test_file = 'data/restaurants_trial_english_sl.xml'

df_train = data_xml_to_df(train_file)
df_test = data_xml_to_df(test_file)

if __name__ == '__main__':
  # df_test = replaceNA(df_test)
  # df_train = replaceNA(df_train)
  # df_test = splitEntityAspect(df_test)
  # df_test = splitEntityAspect(df_test)
  df_test = preprocess(df_test)
  df_train = preprocess(df_train)
  explore(df_train, df_test)
  print(len(df_train), len(df_test))
  df_new_train, df_new_test = process(df_train, df_test)
  print(df_new_train.loc[42, 'text'])
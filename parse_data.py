# -*- encoding: utf-8 -*-
import pandas as pd
import xml.etree.ElementTree as ET
import numpy as np


def data_xml_to_df(xml_file):
  """Converts our raw xml data file into a pandas dataframe。
    
    Args:
      xml_file: xml file path

    Return:
      df: pandas dataframe, contains the extracted data sorted into the
          following columns: opinion_id, text_content, target, category, 
          polarity, ote_start, ote_stop
    
  """

  # parse the xml file
  tree = ET.parse(xml_file)
  root = tree.getroot()

  # initialize variables to populate
  opinion_id = []
  text_content = []
  target = []
  category = []
  polarity = []
  ote_start = []
  ote_stop = []

  # jump directly to the 'sentence' branches and iterate
  for sentence in root.iter('sentence'):

    sentence_id = sentence.get('id')
    sentence_text = sentence.find('text').text
    opinions = sentence.find('Opinions')

    # the number of opinions associated with a sentence varies
    if opinions is None:  # no associated opinions...
      opinion_id.append(sentence_id + ':0')
      text_content.append(sentence_text)
      target.append(np.nan)
      category.append(np.nan)
      polarity.append(np.nan)
      ote_start.append(np.nan)
      ote_stop.append(np.nan)
    else:  # one or more opinions...
      for (i, opinion) in enumerate(sentence.find('Opinions')):
        opinion_id.append(sentence_id + ':%s' % i)
        text_content.append(sentence_text)
        target.append(opinion.get('target'))
        category.append(opinion.get('category'))
        polarity.append(opinion.get('polarity'))
        ote_start.append(opinion.get('from'))
        ote_stop.append(opinion.get('to'))

  # Now convert to dataframe:
  # (column names are specified upfront to define column order)
  df = pd.DataFrame(columns=[
      'opinion_id', 'text', 'target', 'category', 'polarity', 'ote_start',
      'ote_stop'
  ])

  df['opinion_id'] = opinion_id
  df['text'] = text_content
  df['target'] = target
  df['category'] = category
  df['polarity'] = polarity
  df['ote_start'] = ote_start
  df['ote_stop'] = ote_stop

  return df

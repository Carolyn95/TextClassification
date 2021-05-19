import unittest
import sys
sys.path.append('/Users/carolyn/MyGit/NLP/TextClassification/')
from parse_data import data_xml_to_df


class TestParseData(unittest.TestCase):

  def setUp(self):
    self.data_file = "../data/ABSA16_Restaurants_Train_SB1_v2.xml"

  def test_xml2df(self):
    parsed_df = data_xml_to_df(self.data_file)
    all_columns = [c for c in parsed_df.columns]
    parsed_columns = [
        'opinion_id', 'text', 'target', 'category', 'polarity', 'ote_start',
        'ote_stop'
    ]
    self.assertCountEqual(all_columns, parsed_columns)


if __name__ == '__main__':
  unittest.main()

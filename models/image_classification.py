"""
Module to classify images taken from
twitter into male or female
Example usage:
gdp = GenderDataPreprocessing('/Users/edwardcannon/Documents/unilever-2016/genderPrediction/training.zip')
    df_test = pd.read_csv('/Users/edwardcannon/Documents/unilever-2016/influencer_validation/Radox/0168ff50-4cb6-4795-8550-93653414a1ed-Influencer-validation-EmSheldon-EOC21.zip',
                          compression='zip', skiprows=6)
    df_test = df_test[pd.notnull(df_test['Gender'])]
    df_test = df_test[df_test.Gender !='unknown']
    gender_predictor = GenderPredictor(gdp, df_test)
    gender_predictor.train()
    gender_predictor.evaluate()
    gender_predictor.visualize()

"""

import pandas as pd
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

__author__ = 'edwardcannon'


class GenderDataPreprocessing(object):
    """
    Preprocess data frame for gender
    prediction
    """
    def __init__(self, input_file):
        self.inputfile = input_file
        self.input_df = pd.DataFrame()
        self.gender_map = {}
        self.female_count_dict = {}
        self.male_count_dict = {}

    def load_clean_data(self, compression='zip',
                        skiprows=6):
        """
        Loads data and cleans
        :param compression: optional compression
        :param skiprows: optional skip x-rows
        :return:
        """
        df = pd.read_csv(self.inputfile,
                         compression=compression,
                         skiprows=skiprows)
        #filter for gender only
        definite_gender_df = df[(df['Gender'] == 'male') | (df['Gender'] == 'female')]
        self.input_df = definite_gender_df[pd.notnull(definite_gender_df['Avatar'])]
        self.__subset_to_useful_features()
        self.__nlp_enrichment()
        self.gender_map = dict(zip(gdp.input_df['first_name'].tolist(),
                                   gdp.input_df['Gender'].tolist()))

    def __subset_to_useful_features(self):
        #expand features / subset features here!
        self.input_df = self.input_df[['Snippet', 'Full Name',
                                       'Avatar', 'Professions',
                                       'Gender']]

    def __nlp_enrichment(self):
        """
        #Calculate:
        # hashtag usage
        # number of words
        # number of at mentions
        # unique words as percentage [lexical diversity]
        # emoticon usage
        :return:
        """
        tknzr = TweetTokenizer()
        snippet_length = []
        unique_words = []
        hashtags = []
        at_mentions = []
        first_name_list = []
        counter = 0
        self.__word_count_distribution()
        for index, row in self.input_df.iterrows():
            print(100*counter/self.input_df.shape[0])
            tokenxs = tknzr.tokenize(row['Snippet'])
            first_name = self._tokenize_full_name(row)
            first_name_list.append(first_name)
            tokens = []
            hashtag_count = 0
            at_mentions_count = 0
            for token in tokenxs:
                tokens.append(token)
                if token.startswith('#'):
                    hashtag_count += 1
                elif token.startswith('@'):
                    at_mentions_count += 1
            at_mentions.append(at_mentions_count)
            hashtags.append(hashtag_count)
            snippet_length.append(len(tokens))
            unique_words.append(len(set(tokens)))
            counter += 1
        self.input_df['snippet_length'] = snippet_length
        self.input_df['unique_words'] = unique_words
        self.input_df['hashtag_count'] = hashtags
        self.input_df['at_mention_count'] = at_mentions
        self.input_df['first_name'] = first_name_list

    def _tokenize_full_name(self, row):
        """
        Tokenizes full name to pull out
        first name only
        :param row: row in pandas data frame
        :return:
        """
        try:
            return str(row['Full Name']).split(" ")[0].lower()
        except IndexError:
            return "N/A"

    def __word_count_distribution(self):
        """
        Subsets data by gender
        """
        tknzr = TweetTokenizer()
        stop = set(stopwords.words('english'))
        to_remove = {':', '.', ',', '!', '-', "'", '"', '?', '(', ')', '...', '/', '[', ']', '*', '|', '+', '>', '<'}

        def get_distribution(input_df, gender_dict):
            """
            Calculates word distribution of gender
            data frame
            """
            counter = 0
            for index, row in input_df.iterrows():
                print("Gender dist:"+str(100*counter/self.input_df.shape[0]))
                tokenxs = tknzr.tokenize(row['Snippet'])
                for token in tokenxs:
                    token = token.strip()
                    if token not in stop or token not in to_remove:
                        if token in gender_dict:
                            gender_dict[token] += 1
                        else:
                            gender_dict[token] = 1
                    else:
                        pass
                counter += 1

        male_subset = self.input_df[self.input_df['Gender'] == 'male']
        female_subset = self.input_df[self.input_df['Gender'] == 'female']
        get_distribution(male_subset, self.male_count_dict)
        get_distribution(female_subset, self.female_count_dict)
        import operator
        self.male_count_dict = sorted(self.male_count_dict.items(), key=operator.itemgetter(1), reverse=True)
        self.female_count_dict = sorted(self.female_count_dict.items(), key=operator.itemgetter(1), reverse=True)


class GenderPredictor(object):
    """
    Machine learning model for gender prediction
    """
    def __init__(self, gdp, test_dataset):
        self.gdp = gdp
        self.test_df = test_dataset
        self.rf = RandomForestClassifier(n_estimators=100)

    def train(self):
        self.gdp.load_clean_data()
        train_df = self.gdp.input_df.sample(10000)
        self.rf.fit(train_df[['snippet_length',
                              'unique_words',
                              'hashtag_count',
                              'at_mention_count']],
                    train_df[['Gender']])

    def evaluate(self):
        gender_pred = []
        #calculate additional metrics
        tknzr = TweetTokenizer()
        snippet_length = []
        unique_words = []
        hashtags = []
        at_mentions = []
        first_name_list = []
        counter = 0
        for index, row in self.test_df.iterrows():
            print("Evaluating:"+str(100*counter/self.test_df.shape[0]))
            tokenxs = tknzr.tokenize(row['Snippet'])
            tokens = []
            hashtag_count = 0
            at_mentions_count = 0
            for token in tokenxs:
                tokens.append(token)
                if token.startswith('#'):
                    hashtag_count += 1
                elif token.startswith('@'):
                    at_mentions_count += 1
            at_mentions.append(at_mentions_count)
            hashtags.append(hashtag_count)
            snippet_length.append(len(tokens))
            unique_words.append(len(set(tokens)))
            counter += 1
        self.test_df['snippet_length'] = snippet_length
        self.test_df['unique_words'] = unique_words
        self.test_df['hashtag_count'] = hashtags
        self.test_df['at_mention_count'] = at_mentions
        countx = 0
        for index, row in self.test_df.iterrows():
            fname = str(row['Full Name']).split(" ")[0].lower()
            if fname in self.gdp.gender_map:
                gender = self.gdp.gender_map[fname]
                gender_pred.append(gender)
                countx += 1
            else:
                result = self.rf.predict(row[['snippet_length',
                                              'unique_words',
                                              'hashtag_count',
                                              'at_mention_count']])
                result = str(result).replace("['", "").replace("']", "")
                gender_pred.append(result)
                print('Need to predict gender based on attributes other '
                      'than first name!')
        self.test_df['gender_pred'] = gender_pred

    def visualize(self):
        dx = self.test_df[['gender_pred', 'Gender']]
        dx = dx.replace('female', 0)
        dx = dx.replace('male', 1)
        fpr, tpr, thresholds = roc_curve(dx['Gender'], dx['gender_pred'])
        roc_auc = auc(fpr, tpr)
        plt.title('Receiver Operating Characteristic')
        plt.plot(fpr, tpr, 'b', label='AUC = %0.2f'% roc_auc)
        plt.legend(loc='lower right')
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlim([-0.1, 1.2])
        plt.ylim([-0.1, 1.2])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.show()


if __name__ == '__main__':
    pass
from builtins import list, len, map, int, object, min, max, str, zip, dict

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MultiLabelBinarizer
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

class PreProccessing():
    regression = None
    def __init__(self, regDataset, classDataset, regression=True):
        if regression:
            self.regression = True
            self.moviesFile = pd.read_csv(regDataset) #'Movies_training.csv')
        else:
            self.regression = False
            self.moviesFile = pd.read_csv(classDataset) #'Movies_training_classification.csv')
        self.moviesFile.drop(['Type'], axis=1, inplace=True)
        self.dataPreproccessing()

    def dataPreproccessing(self):
        # Dropping the cols missing 50% and more
        # nulls = self.check_null(50)
        cols = ['Directors', 'Genres', 'Country', 'Language', 'Runtime', 'Year', 'Age', 'Rotten Tomatoes']  #hulu , prime , disney
        str_pred= 'rate'
        if self.regression:
           str_pred ='IMDb'

        cols.append(str_pred)
        self.Replace_Missed(cols)
        self.moviesFile['Rotten Tomatoes']= list(map(int, self.moviesFile['Rotten Tomatoes'].str[:-1]))
        self.moviesFile.Year = self.moviesFile.Year.astype("object")

        if self.regression:
            self.applyMultiLabelBinarizer(['Genres'])
            self.applyLabelEncoding(['Country', 'Language', 'Year', 'Directors', 'Age'])
            self.normalization(
                ['Year', 'Runtime', str_pred, 'Directors', 'Age', 'Rotten Tomatoes', 'Country', 'Language'])

        else:
            self.applyMultiLabelBinarizer(['Genres'])
            self.applyLabelEncoding(['Country', 'Language','Year', 'Directors', 'Age'])
            self.applyLabelEncoding([str_pred])
            self.normalization(
                ['Year', 'Runtime', str_pred, 'Directors', 'Age', 'Rotten Tomatoes', 'Country', 'Language'])

        self.moviesFile.drop(['Title'], axis=1, inplace=True)

        tmp = self.moviesFile.pop(str_pred)           #move label column to end of dataframe
        self.moviesFile[str_pred]= tmp
    '''
        self.applyOneHotEncoding(['Language', 'Year', 'Directors', 'Genres', 'Country', 'Age'])
        if not self.regression:
            self.applyOneHotEncoding([str_pred])
        #self.normalization(
        #    ['Year', 'Runtime', str_pred, 'Language', 'Directors', 'Genres', 'Country', 'Age', 'Rotten Tomatoes'])
        self.moviesFile.drop(['Title'], axis=1, inplace=True)
        tmp = self.moviesFile.pop(str_pred)  # move label column to end of dataframe
        self.moviesFile[str_pred] = tmp
    '''

    def get_Average(self, idx):
        return self.moviesFile[idx].sum() /len(self.moviesFile[idx])
    def get_Mean(self, idx):
        return self.moviesFile[idx].mean()

    def get_Median(self, idx):
        return  self.moviesFile[idx].median()

    def Replace_Missed(self, cols):
        for col in cols:
            if(self.moviesFile[col].dtype == object):
                most_frq_row=str(self.moviesFile[col].value_counts()[self.moviesFile[col].value_counts() == self.moviesFile[col].value_counts().max()])
                self.moviesFile[col]= self.moviesFile[col].replace(np.nan, most_frq_row.split('   ')[0])
            else:
                self.moviesFile[col]= self.moviesFile[col].replace(np.nan, self.get_Median(col))
        return

    def applyMultiLabelBinarizer(self, columns_bridge):
        for idx in columns_bridge:
            bridge_df = pd.DataFrame(self.moviesFile, columns=[idx])
            bridge_df[idx] = [x.split(',') for x in bridge_df[idx]]
            mlb = MultiLabelBinarizer()
            enc_df = pd.DataFrame(mlb.fit_transform(bridge_df[idx]))
            enc_df.columns = mlb.classes_      #rename columns
            self.moviesFile = self.moviesFile.join(enc_df)
            self.moviesFile.drop([idx], axis=1, inplace=True)

    def applyLabelEncoding(self, columns_bridge):
        for idx in columns_bridge:
            bridge_df = pd.DataFrame(self.moviesFile, columns=[idx])
            PreProccessing.Label_encoder = LabelEncoder()
            bridge_df[idx] = PreProccessing.Label_encoder.fit_transform(bridge_df[idx])
            self.moviesFile[idx] = bridge_df[idx]


####################################################################################3
    def applyOneHotEncoding(self, columns_bridge):
        for idx in columns_bridge:
            bridge_df = self.OneHotEncoding(idx)
            bridge_df.drop([idx + "_ID"], axis=1, inplace=True)
            bridge_df.drop([idx], axis=1, inplace=True)
            self.moviesFile.join(bridge_df)
        print(bridge_df)

    def OneHotEncoding(self, col_bridge):
        bridge_df = pd.DataFrame(self.moviesFile, columns=[col_bridge])
        bridge_df = bridge_df.rename(columns={0: col_bridge + "_ID"})
        PreProccessing.Label_encoder = LabelEncoder()
        bridge_df[col_bridge + "_ID"] = PreProccessing.Label_encoder.fit_transform(bridge_df[col_bridge])
        one_hot_encoder = OneHotEncoder(handle_unknown='ignore', categories='auto')
        enc_df = pd.DataFrame(one_hot_encoder.fit_transform(bridge_df[[col_bridge + "_ID"]]).toarray())
        #enc_df.columns = one_hot_encoder.
        bridge_df = bridge_df.join(enc_df)
        return bridge_df

    def normalization(self, cols):
        for i in cols:
            self.moviesFile[i] = (self.moviesFile[i] - min(self.moviesFile[i])) / (max(self.moviesFile[i]) - min(self.moviesFile[i]))

    def show_data(self, X, y, x_label='X', y_label='Y', title='Data'):
        '''
        Take to columns and draw them
        :param X: the x-axis column
        :param y: the y-axis column
        :return: visual data plot
        '''
        plt.scatter(X, y)
        plt.title = title
        plt.xlabel(x_label, fontsize=20)
        plt.ylabel(y_label, fontsize=20)
        plt.show()



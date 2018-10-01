# -*- coding: utf-8 -*-

# import packages
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import jieba.posseg as pseg
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score



# define data path
dataset_path = '../data'
datafile = os.path.join(dataset_path, 'DMSC.csv')

# define stop word path
stop_words_path = '../stop_words'

# load stopword
stopwords1 = [line.rstrip() for line in open(os.path.join(stop_words_path, '中文停用词库.txt'), 'r',
                                             encoding='utf-8')]
stopwords2 = [line.rstrip() for line in open(os.path.join(stop_words_path, '哈工大停用词表.txt'), 'r',
                                             encoding='utf-8')]
stopwords3 = [line.rstrip() for line in
              open(os.path.join(stop_words_path, '四川大学机器智能实验室停用词库.txt'), 'r', encoding='utf-8')]
stopwords = stopwords1 + stopwords2 + stopwords3


def proc_text(raw_line):
    """
        processing text
    """

    
    filter_pattern = re.compile('[^\u4E00-\u9FD5]+')
    chinese_only = filter_pattern.sub('', raw_line)


    word_list = pseg.cut(chinese_only)


    used_flags = ['v', 'a', 'ad']
    meaninful_words = []
    for word, flag in word_list:
        #
        if (word not in stopwords) and (flag in used_flags):
            meaninful_words.append(word)
    return ' '.join(meaninful_words)


def main():
    """
        main function
    """
    # load data
    raw_data = pd.read_csv(datafile)

    # task1. check the data
    print('\n===================== task1. check the data =====================')
    print('The total dataset has {} records.'.format(len(raw_data)))

    # movie names
    print('The dataset has {} movies'.format(len(raw_data['Movie_Name_CN'].unique())))
    print(raw_data['Movie_Name_CN'].unique())

    # caculating the mean score for movie
    movie_mean_score = raw_data.groupby('Movie_Name_CN')['Star'].mean().sort_values(ascending=False)
    movie_mean_score.plot(kind='bar')
    plt.tight_layout()

    # task2. preprocessing tha data
    # drop missing data
    cln_data = raw_data.dropna().copy()

    # creating a new feature, if the rating >=3.0，we rated it as positive label 1,
    #else rated as negative and label it as 0
    cln_data['Positively Rated'] = np.where(cln_data['Star'] >= 3, 1, 0)

    # checking the head of the data
    print('check the head of data：')
    print(cln_data.head())

    
    cln_data['Words'] = cln_data['Comment'].apply(proc_text)

    print('checking the data again：')
    print(cln_data.head())

    # save the processed data
    saved_data = cln_data[['Words', 'Positively Rated']].copy()
    saved_data.dropna(subset=['Words'], inplace=True)
    saved_data.to_csv(os.path.join(dataset_path, 'douban_cln_data.csv'), encoding='utf-8', index=False)

    # split training data and test data
    from sklearn.model_selection import train_test_split

    X_train_data, X_test_data, y_train, y_test = train_test_split(saved_data['Words'], saved_data['Positively Rated'],
                                                                  test_size=1 / 4, random_state=0)

    # Task3. text character
    print('\n===================== Task3. text character  =====================')

    # use max_features to get the top frequent word 
    n_dim = 10000
    vectorizer = TfidfVectorizer(max_features=n_dim)
    X_train = vectorizer.fit_transform(X_train_data.values)
    X_test = vectorizer.transform(X_test_data.values)

    print('vector dimension:', len(vectorizer.get_feature_names()))
    print('top word: {}:'.format(n_dim))
    print(vectorizer.get_feature_names())

    # task4. build the model and make prediction
    print('\n===================== Task4. build the model and make prediction=====================')
    lr_model = LogisticRegression(C=100)
    lr_model.fit(X_train, y_train)

    predictions = lr_model.predict(X_test)
    print('AUC:', roc_auc_score(y_test, predictions))


if __name__ == '__main__':
    main()

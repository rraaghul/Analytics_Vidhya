import os
from keras.models import Sequential
from keras.layers import Dense,Dropout
import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
stop_words = list(set(stopwords.words('english')))+['']

df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')
y_train = np.array(df_train['label'].tolist())
test_id = df_test['id'].tolist()
word_list = []

def clean(tweet):
    filtered_sentence = []
    tweet = re.sub('[^\x00-\x7F]','',tweet)
    tweet = re.sub('@|#|%|&|\(|\)|\'|!|~|\/|\.|\[|\]|\?|\:|\-|\"|\$|\*','',tweet)
    tweet = re.sub('\d+','',tweet)
    word_tokens = re.split(' |,',tweet)
    filtered_sentence = [lemmatizer.lemmatize(w.lower()) for w in word_tokens if not w in stop_words and not w.isdigit()]
    word_list.extend(filtered_sentence)    
    return ','.join(filtered_sentence)

df_train['clean_tags'] = df_train['tweet'].apply(lambda x:clean(x))
unique_word_list = list(set(word_list))
df_test['clean_tags'] = df_test['tweet'].apply(lambda x:clean(x))

tfidf = TfidfVectorizer(vocabulary = unique_word_list,min_df = 1)
X_train = tfidf.fit_transform(df_train['clean_tags'].tolist())
# Transform a document into TfIdf coordinates
X_test = tfidf.transform(df_test['clean_tags'].tolist())



def create_model():
# create model
    model = Sequential()
    model.add(Dense(512, input_dim = X_train.shape[1], kernel_initializer = 'normal', activation = 'relu'))
    model.add(Dropout(0.1))
    model.add(Dense(128,kernel_initializer = 'normal', activation = 'relu'))
    model.add(Dense(1, kernel_initializer = 'normal', activation = 'sigmoid'))
    # Compile model
    model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    model.summary()
    return model


model = create_model()
model.fit(X_train,y_train,batch_size = 64,epochs = 3)

y_test = model.predict_classes(X_test)

submit = pd.DataFrame({'id': test_id, 'label': y_test.reshape(-1)})
submit.to_csv('output.csv',index = False)

import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import *
from keras.callbacks import EarlyStopping
from keras.models import load_model
from sklearn.model_selection import train_test_split
import re


def load_file(filename):
    data = pd.DataFrame()
    filetype = filename.split(".")[-1]

    # Parsing txt file
    if filetype == "txt":
        parsed_data = {'comment_text': [], 'label': []}
        with open(filename, 'rt') as data_in:
            for line in data_in:
                # Skip first line
                if not line.lower().startswith("tweet index"):
                    line = line.rstrip()
                    label = int(line.split("\t")[1].replace("0", "-1"))
                    tweet = line.split("\t")[2]
                    parsed_data['comment_text'].append(tweet)
                    parsed_data['label'].append(label)

        data = pd.DataFrame(data=parsed_data)
        for idx, row in data.iterrows():
            row[0] = row[0].replace('rt', ' ')

    # Parsing csv file with panda
    elif filetype == "csv":
        # Importing CSV from Kaeggle, Labled -1 for non ironic and 1 for ironic text
        data = pd.read_csv('data/irony/kaggle-irony-labeled.csv')
        data = data[['comment_text', 'label']]  # Headline in CSV Document

    # simple preprocessing
    data['comment_text'] = data['comment_text'].apply(lambda x: x.lower())
    data['comment_text'] = data['comment_text'].apply((lambda x: re.sub('[^a-zA-z0-9\s]', '', x)))

    ironic_posts = 0
    non_ironic_posts = 0
    for label in data['label']:
        if label == 1:
            ironic_posts += 1
        else:
            non_ironic_posts += 1

    print("Non ironic posts: " + str(non_ironic_posts))
    print("Ironic posts: " + str(ironic_posts))

    return data


dataset_kaggle = load_file("data/irony/kaggle-irony-labeled.csv")
print(dataset_kaggle)
dataset_semeval = load_file("data/irony/SemEval2018-T3-train-taskA.txt")
print(dataset_semeval)


def splitdata(data, max_features=2000):
    # Tokenizing the data and splitting the dataset
    # Generating Train, Test and Dev set 80 / 10 / 10
    tokenizer = Tokenizer(num_words=max_features, split=' ')
    tokenizer.fit_on_texts(data['comment_text'].values)
    tokenized_data_x = tokenizer.texts_to_sequences(data['comment_text'].values)
    tokenized_data_x = pad_sequences(tokenized_data_x)
    Y = pd.get_dummies(data['label']).values

    X_train, X_test, Y_train, Y_test = train_test_split(tokenized_data_x, Y, test_size=0.20, random_state=42)
    validation_size = int(len(X_test) / 2)
    X_validate = X_test[-validation_size:]
    Y_validate = Y_test[-validation_size:]
    X_test = X_test[:-validation_size]
    Y_test = Y_test[:-validation_size]
    return X_train, Y_train, X_validate, Y_validate, X_test, Y_test


def train_model(name, dataset, pretrain_model=None):
    X_train, Y_train, X_validate_kaggle, Y_validate_kaggle, X_test, Y_test = splitdata(dataset)

    max_fatures = 2000
    lstm_out = 196
    embed_dim = 128
    dropout_value = 0.2

    earlyStopping = EarlyStopping(monitor='loss', min_delta=0, patience=2, verbose=0, mode='auto')
    model = Sequential()
    model.add(Embedding(max_fatures, embed_dim))
    model.add(SpatialDropout1D(dropout_value))
    model.add(LSTM(lstm_out, dropout=dropout_value, recurrent_dropout=dropout_value))
    model.add(Dense(2, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])
    if pretrain_model is not None:
        model.set_weights(load_model("data/saved_models/" + pretrain_model + ".h5").get_weights())
        name = "pretrained_" + name

    print(model.summary())
    model.fit(X_train, Y_train, epochs=100, batch_size=32, verbose=2, callbacks=[earlyStopping])
    loss, acc = model.evaluate(X_test, Y_test, verbose=2, batch_size=32)

    print("Evaluated on Testset: Loss: %.2f" % loss)
    print("Evaluated on Testset: Acc: %.2f" % acc)
    model.save('data/saved_models/' + name + '.h5')


train_model("kaggle", dataset_kaggle)
train_model("semeval", dataset_semeval)
train_model("kaggle", dataset_kaggle, "semeval")
train_model("semeval", dataset_semeval, "kaggle")

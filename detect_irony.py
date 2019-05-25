import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import *
from keras.callbacks import EarlyStopping
from keras.models import load_model
from sklearn.model_selection import train_test_split
import re

# Task: Predict whether comments are ironic.
# How can one domain (large data set) help prediction on
# another domain (smaller dataset)?
# Pretrain with data A, continue training with data B
# Using Kaggle Dataset and Semeval Dataset. Use readme file for direct links


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
# Non ironic posts: 1412
# Ironic posts: 537
#                                            comment_text  label
# 0     i suspect atheists are projecting their desire...     -1
# 1     its funny how the arguments the shills are mak...     -1
# 2     we are truly following the patterns of how the...     -1
# 3     air pressure dropping as altitude goes higher ...     -1
# 4     absolutely  i think wed be hard pressed to fin...     -1
# 5     democrats dont know how to manage money   shoc...      1
# 6     its not like automation has eliminated the nee...      1
# 7                                          would he win     -1
# 8     yeah i didnt get far  this article fills me wi...     -1
# 9     insane like a fox  ted cruz is actually very v...      1
# 10    what kind of heartless person down voted this ...     -1
# 11    if jesus destroyed satan for our freedom why i...      1
# 12    god i hope so  and when it does i will keep ki...     -1
# 13    because you cant work the part timers crazy ho...     -1
# 14    they only help the poor to promote their god t...     -1
# 15    gt the essence of the entitlement state is gov...     -1
# 16    because the reddit liberal brigade would downv...      1
# 17    its amazing how democrats view money  it has t...      1
# 18    the tides of change are on the move  legalizin...     -1
# 19    to be fair there are many who are only bigots ...     -1
# 20    that just makes me cringe   not sure if its th...      1
# 21    i used to pirate videogames then steam came al...     -1
# 22    young single guy here  my policy was perfect f...     -1
# 23    gt2    churches are antiscience \n\ngtdespite ...     -1
# 24    the only people offended by the redskins are w...     -1
# 25    so the question during their twitter qampa abo...      1
# 26    i havent read the article yet and i have absol...     -1
# 27    this really is pathetic \n\nyou send shit to o...      1
# 28    i for one am in utter shock that these fine in...      1
# 29                                 im sure it will pass      1
# ...                                                 ...    ...
# 1919  welcome to the new congress same as the old co...     -1
# 1920  it was still an incredibly lucrative investmen...     -1
# 1921  so apparently catholics arent christians accor...     -1
# 1922  could you image how poor of a state they would...     -1
# 1923  the most amazing part of that episode was when...     -1
# 1924  why only spending cuts  include subsidy slashe...     -1
# 1925  im going to use this comment without giving cr...      1
# 1926  because its texas  the place where they slip c...     -1
# 1927  i have a feeling the aca will be that little t...     -1
# 1928  utah  our jesus can beat up your jesus \n\ncan...      1
# 1929          gt secrete rulings\n\nthat explains a lot     -1
# 1930  i thought this was even more impressive\n\ngt ...     -1
# 1931  benefits\n\n fewer drug enforcement officers n...     -1
# 1932  check out the comment section on this video on...     -1
# 1933  single payer healthcare works great where i li...     -1
# 1934  i prefer it  im a texan and a while back i had...      1
# 1935  blacks have gone full stupid with the race car...     -1
# 1936  this would make me so happy  it sounds like th...     -1
# 1937  drivethru abortions are next \n\n ive driven h...      1
# 1938  while dick is an ass hat it is generally consi...     -1
# 1939  moore is a big supporter of veterans   he runs...      1
# 1940  something i read earlier today\n\ngt  conserva...     -1
# 1941  this was what folks didnt understand about the...     -1
# 1942    at least the new york post is telling the truth     -1
# 1943  edit ill preface with thisthe title and articl...     -1
# 1944  this is an interesting point  there are no sho...     -1
# 1945  maybe you mean how we are to respond to the go...     -1
# 1946                     [ what ]httptinyurl comkw5cpxz     -1
# 1947  does anybody remember during one of the debate...     -1
# 1948  the pope is meeting a cruel dictator   likely ...      1
#
# [1949 rows x 2 columns]
# Non ironic posts: 1923
# Ironic posts: 1911
#                                            comment_text  label
# 0     sweet united nations video just in time for ch...      1
# 1     mrdahl87 we are rumored to have talked to ervs...      1
# 2     hey there nice to see you minnesotand winter w...      1
# 3                    3 episodes left im dying over here     -1
# 4     i cant breathe was chosen as the most notable ...      1
# 5     youre never too old for footie pajamas httptco...     -1
# 6     nothing makes me happier then getting on the h...      1
# 7     430 an opening my first beer now gonna be a lo...     -1
# 8     adam_klug do you think you would support a guy...     -1
# 9     samcguigan544 you are not allowed to open that...     -1
# 10    oh thank god  our entire office email system i...      1
# 11    but instead im scrolling through facebook inst...     -1
# 12    targetzonept pouting_face no he bloody isnt i ...     -1
# 13    cold or warmth both suffuse ones cheeks with p...     -1
# 14    just great when youre mobile bill arrives by text      1
# 15    crushes are great until you realize theyll nev...      1
# 16    buffalo sports media is smarter than all of us...      1
# 17    i guess my cat also lost 3 pounds when she wen...     -1
# 18    yankeeswfan ken_rosenthal trading a sp for a d...      1
# 19    but darklightdave was trying to find us and my...      1
# 20    deputymartinski please doi need the second han...      1
# 21    i never cared for beyonce bc i could never get...     -1
# 22                 ywtorres9 time to hit the books then      1
# 23    rushordertees thx4flw flwthemusic elektrikeven...     -1
# 24    love these cold winter mornings grimacing_face...      1
# 25    amazingly httptconeiozunbld is not owned by bh...     -1
# 26    wish she could have told me herself nicolesche...     -1
# 27    the rain has made extra extra lazysmiling_face...     -1
# 28    i was doing great with this summary of my year...     -1
# 29    see that might show up on a background check a...     -1
# ...                                                 ...    ...
# 3804  nylons quick13 jamieyuccas chadhartman it was ...     -1
# 3805  thefollowingfox i get paid 4 posting stuff lik...     -1
# 3806  abelv03 kwapt i just want learning from this g...     -1
# 3807  only ones in the cinema  putting my phone on s...     -1
# 3808  bbcradmac stuartmaconie years ago in m  s in r...      1
# 3809  montana of 300 14  the best versace remix in t...     -1
# 3810  i should of just made a canvas of coffee stain...      1
# 3811      the world is such a smiley place flushed_face      1
# 3812  two broke rednecks fatherdaughter riffing team...     -1
# 3813  wtf is happening to these kids are you kidding...     -1
# 3814  i would have made a much more convincing bella...     -1
# 3815  i retweeted this so chris graham blocked me fa...      1
# 3816             fries with that 304 alabamastatemajors     -1
# 3817  startupgrindbuf magnachef if you need a dev en...     -1
# 3818  im glad the dc council has its priorities inta...      1
# 3819             riding the distraction train choo choo     -1
# 3820  chill repost dead  dominos haha face_with_tear...     -1
# 3821  someone i work w doesnt let his kids believe i...      1
# 3822  check out my new post myfairdaily 10 things iv...     -1
# 3823  obama whisked away to hospital diagnosed with ...      1
# 3824  dcsportsgrl dragonflyjonez true n thats y we r...     -1
# 3825  another one of our support vehicles modified f...     -1
# 3826                  thanks for shutting the city down      1
# 3827  flippysgardenia ikr dont you see hes gonna cry...      1
# 3828  glad theres not a typhoon where we go on holid...      1
# 3829   banditelli regarding what the psu president does     -1
# 3830  banditelli but still bothers me that i see now...     -1
# 3831  well now that ive listened to all of into the ...     -1
# 3832  hummingbirds are  experts at hovering after al...     -1
# 3833  only thing missing now is a session at the gym...     -1
#
# [3834 rows x 2 columns]

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

# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# embedding_1 (Embedding)      (None, None, 128)         256000
# _________________________________________________________________
# spatial_dropout1d_1 (Spatial (None, None, 128)         0
# _________________________________________________________________
# lstm_1 (LSTM)                (None, 196)               254800
# _________________________________________________________________
# dense_1 (Dense)              (None, 2)                 394
# =================================================================
# Total params: 511,194
# Trainable params: 511,194
# Non-trainable params: 0
# _________________________________________________________________

train_model("kaggle", dataset_kaggle)
train_model("semeval", dataset_semeval)
train_model("kaggle", dataset_kaggle, "semeval")
train_model("semeval", dataset_semeval, "kaggle")

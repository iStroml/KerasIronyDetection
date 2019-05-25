import re
import pandas as pd


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
        if label == 1: ironic_posts += 1
        else: non_ironic_posts += 1


    print("Non ironic posts: " + str(non_ironic_posts))
    print("Ironic posts: " + str(ironic_posts))

    return data

dataset_kaggle = load_file("data/irony/kaggle-irony-labeled.csv")
print(dataset_kaggle)
dataset_semeval = load_file("data/irony/SemEval2018-T3-train-taskA.txt")
print(dataset_semeval)
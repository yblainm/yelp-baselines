from nltk import word_tokenize
import re


def prepare_data(text):
    """

    :param text: raw text file
    :type text: str
    """
    lines = []
    labels = []
    for line in text.splitlines():
        if len(line) is not 0:
            lines.append(prepare_text(line[:-2]))   # Everything except the tab
            labels.append(int(line[-1]))            # The label
    return lines, labels


def prepare_text(text):
    return word_tokenize(re.sub(r'\W', ' ', text.lower().replace('\'','')))


def datafromfile(path):
    with open(path, mode="r", encoding="utf8") as f:
        text = f.read()
        f.close()
    return prepare_data(text)


yelp_train_x, yelp_train_y = datafromfile("yelp-train.txt")

# print(*lines[:5], sep='\n')
# print(labels[:5])

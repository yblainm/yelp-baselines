from nltk import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from collections import defaultdict
import re, numpy as np, os


def prepare_data(text):
    """
    :param text: raw text file
    :type text: str
    :return list of str lines, list of int labels
    """
    raw = []    # Raw lines
    lines = []  # Tokenized
    labels = [] # Labels
    for line in text.splitlines():
        if len(line) is not 0:
            raw.append(prepare_text(line[:-2]))     # :-2 to exclude \t
            lines.append(prepare_text(line[:-2], token=True))
            labels.append(int(line[-1]))            # The label
    return raw, lines, labels


def prepare_text(text, token=False):
    return word_tokenize(re.sub(r'\W', ' ', text.lower().replace('\'',''))) if token else re.sub(r'\W', ' ', text.lower().replace('\'',''))


def datafromfile(path):
    with open(path, mode="r", encoding="utf8") as f:
        text = f.read()
        f.close()
    return prepare_data(text)


def generate_vocabulary(corpus):
    """
    :param corpus: list of lists of word token strings (from prepare_data)
    :type list
    :return list of tuples (str word, int count)
    """

    # Equivalently, CountVectorizer with max_features makes this useless

    countsd = defaultdict(int)

    for x in corpus:
        for token in x:
            countsd[token] = countsd[token] + 1

    counts = [(token, count) for token, count in countsd.items()]
    counts.sort(key=lambda x: x[1], reverse=True)
    counts = counts[:10000]
    return counts


def export_vocab(fname, vocab):
    out = ""
    for i in range(0, len(vocab)):
        out += vocab[i][0]  # Word type
        out += "\t"+str(i)  # numeric id
        out += "\t"+str(vocab[i][1])+"\n"   # frequency
    with open(fname, "w", encoding="utf8") as f:
        f.write(out)
        f.close()


def export_data(fname, x_tokens, vectorizer, y):
    out = ""
    for i in range(0, len(y)):
        for token in x_tokens[i]:
            indices = np.nonzero(vectorizer.transform(np.array([token])))
            # print(indices[1][0] if indices[0].shape[0] != 0 else "")
            out += (str(indices[1][0]) if indices[0].shape[0] != 0 else "")
            out += " "
        out = out[:-1] + "\t" + str(y[i]) + "\n"
    with open(fname, mode="w", encoding="utf8") as f:
        f.write(out)
        f.close()


# YELP
yelp_train_raw, yelp_train_tokens, yelp_train_y = datafromfile(os.path.join("data","yelp-train.txt"))
# yelp_vocab = generate_vocabulary(yelp_train_tokens) # For exported DB
# export_vocab(os.path.join("out","yelp-vocab.txt"), yelp_vocab)

yelp_vectorizer = CountVectorizer(max_features=10000)
yelp_train_x = yelp_vectorizer.fit_transform(np.array(yelp_train_raw))
# export_data(os.path.join("out","yelp-train.txt"), yelp_train_tokens, yelp_vectorizer, yelp_train_y)
# export_data(os.path.join("out","data-export-test.txt"), [["this", "is", "a", "test"]], yelp_vectorizer, [3])
# print(np.nonzero(yelp_vectorizer.transform(["test"]))[1][0])

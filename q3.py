from nltk import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from collections import defaultdict
import re, numpy as np, os


def prepare_data(path):
	"""
	:param text: raw text file
	:type text: str
	:return list of str lines, list of int labels
	"""
	raw = []    # Raw lines
	lines = []  # Tokenized
	labels = [] # Labels
	with open(path, mode="r", encoding="utf-8") as f:
		text = f.readlines()
		for line in text:
			if len(line) > 0:
				raw.append(prepare_text(line[:-3]))     # :-2 to exclude \t
				lines.append(prepare_text(line[:-3], token=True))
				labels.append(int(line[-2]))            # The label
	return raw, lines, labels


def prepare_text(text, token=False):
	return word_tokenize(re.sub(r'\W', ' ', text.lower().replace('\'',''))) \
		if token else re.sub(r'\W', ' ', text.lower().replace('\'',''))


def datafromfile(path):
	with open(path, mode="r", encoding="utf-8") as f:
		text = f.read()
		f.close()

	raw, lines, labels = prepare_data(path)
	return text, raw, lines, labels


def export_vocabulary(path, raw, vectorizer):
	"""

	:param raw: entire raw corpus text string
	:param vectorizer: pre-fit vectorizer
	:type vectorizer CountVectorizer
	:type raw list
	"""

	x = vectorizer.transform(raw)
	# [word, id, frequency]
	featurenames = vectorizer.get_feature_names()
	indices = { k : featurenames.index(k) for k in vectorizer.vocabulary_.keys()}
	vocab = [k+" "+str(indices[k])+" "+str(x[0,indices[k]])+"\n" for k,v in vectorizer.vocabulary_.items()]
	with open(path, "w") as f:
		f.writelines(vocab)
		f.close()
	return vocab


def export_data(fname, x_tokens, vectorizer, y):
	'''

	:param vectorizer:
	:type vectorizer CountVectorizer
	:return:
	'''
	out = ""
	for i in range(0, len(y)):
		for token in x_tokens[i]:
			idx = vectorizer.vocabulary_.get(token)
			if idx:
				out += str(vectorizer.vocabulary_[token])
				out += " "
		out = out[:-1] + "\t" + str(y[i]) + "\n"
	with open(fname, mode="w", encoding="utf-8") as f:
		f.write(out)
		f.close()

# with open(os.path.join("data","yelp-train.txt"), encoding="utf8") as f:
# 	s = f.readlines()
# 	for line in s:
# 		for i in range(0,len(line)):
# 			# if line[i] == '\u0085':
# 			print(i, line[-2])
#
# quit()

# YELP
	# Prepare data
yelp_train_raw, yelp_train_rawlines, yelp_train_tokenlines, yelp_train_y = \
	datafromfile(os.path.join("data","yelp-train.txt"))
yelp_valid_raw, yelp_valid_rawlines, yelp_valid_tokenlines, yelp_valid_y = \
	datafromfile(os.path.join("data","yelp-valid.txt"))
yelp_test_raw, yelp_test_rawlines, yelp_test_tokenlines, yelp_test_y = \
	datafromfile(os.path.join("data","yelp-test.txt"))

yelp_vectorizer = CountVectorizer(max_features=10000)
yelp_train_x = yelp_vectorizer.fit_transform(np.array(yelp_train_rawlines))
yelp_valid_x = yelp_vectorizer.transform(np.array(yelp_valid_rawlines))
yelp_test_x = yelp_vectorizer.transform(np.array(yelp_test_rawlines))
yelp_train_y = np.array(yelp_train_y)
yelp_valid_y = np.array(yelp_valid_y)
yelp_test_y = np.array(yelp_test_y)


	# Export data as per instructions
print(export_vocabulary(os.path.join("out","yelp-vocab.txt"),
						[prepare_text(yelp_train_raw, False)], yelp_vectorizer))
export_data(os.path.join("out","yelp-train.txt"), yelp_train_tokenlines, yelp_vectorizer, yelp_train_y)
export_data(os.path.join("out","yelp-valid.txt"), yelp_valid_tokenlines, yelp_vectorizer, yelp_valid_y)
export_data(os.path.join("out","yelp-test.txt"), yelp_test_tokenlines, yelp_vectorizer, yelp_test_y)

# IMDB
	# Prepare data
IMDB_train_raw, IMDB_train_rawlines, IMDB_train_tokenlines, IMDB_train_y = \
	datafromfile(os.path.join("data","IMDB-train.txt"))
IMDB_valid_raw, IMDB_valid_rawlines, IMDB_valid_tokenlines, IMDB_valid_y = \
	datafromfile(os.path.join("data","IMDB-valid.txt"))
IMDB_test_raw, IMDB_test_rawlines, IMDB_test_tokenlines, IMDB_test_y = \
	datafromfile(os.path.join("data","IMDB-test.txt"))

IMDB_vectorizer = CountVectorizer(max_features=10000)
IMDB_train_x = IMDB_vectorizer.fit_transform(np.array(IMDB_train_rawlines))
IMDB_valid_x = IMDB_vectorizer.transform(np.array(IMDB_valid_rawlines))
IMDB_test_x = IMDB_vectorizer.transform(np.array(IMDB_test_rawlines))
IMDB_train_y = np.array(IMDB_train_y)
IMDB_valid_y = np.array(IMDB_valid_y)
IMDB_test_y = np.array(IMDB_test_y)


	# Export data as per instructions
print(export_vocabulary(os.path.join("out","IMDB-vocab.txt"),
						[prepare_text(IMDB_train_raw, False)], IMDB_vectorizer))
export_data(os.path.join("out","IMDB-train.txt"), IMDB_train_tokenlines, IMDB_vectorizer, IMDB_train_y)
export_data(os.path.join("out","IMDB-valid.txt"), IMDB_valid_tokenlines, IMDB_vectorizer, IMDB_valid_y)
export_data(os.path.join("out","IMDB-test.txt"), IMDB_test_tokenlines, IMDB_vectorizer, IMDB_test_y)

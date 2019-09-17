import spacy
import os
import json
import string

from params import annotation_file, weights_dir

nlp = spacy.load("en_core_web_lg")

pos_dict_weights = {
    'NOUN': 1.1,
    'VERB': 1.05,
    'ADJ': 0.9,
    'ADV': 0.9,
    'SCONJ': 0.8,
    'INTJ': 0.8,
    'DET': 0.8,
    'CCONJ': 0.8,
    'CONJ': 0.8,
    'AUX': 0.8,
}

with open(annotation_file, 'rt') as f:
    data = json.load(f)

all_captions = [x['caption'] for x in data['annotations']]

all_words = set()
for x in all_captions:
    # [all_words.add(item) for item in x.lower().translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation))).strip().split()]
    [all_words.add(item) for item in x.lower().translate(str.maketrans('!"#$%&()*+.,-/:;=?@[\\]^_`{|}~ ', ' ' * len('!"#$%&()*+.,-/:;=?@[\\]^_`{|}~ '))).strip().split()]

weights = {}
for word in all_words:
    doc = nlp(word)
    word_pos = doc[0].pos_

    weights[word] = pos_dict_weights.get(word_pos, 1)

with open(os.path.join(weights_dir, 'tag2score_pos_weight.json'), 'wt') as f:
    json.dump(weights, f)
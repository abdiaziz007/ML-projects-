import numpy as np
import pandas as pd
from collections import defaultdict

from nltk.metrics.scores import precision, recall, f_measure, accuracy
from nltk.tag import untag

from sklearn import metrics

from spacy.tokens import Doc
from spacy.gold import GoldParse
from spacy.scorer import Scorer


def compare_taggers(tagger_dict, test_sentences):
    """Returns DataFrame with metrics for NLTK taggers in `tagger_dict` evaluated on `test_sentences`"""
    compare_dict = dict()
    golds = [tag for sentence in test_sentences for _, tag in sentence]
    
    for name, tagger in tagger_dict.items():
        tagger_preds = [tagger.tag([word for word,_ in test_sentence]) for test_sentence in test_sentences]
        preds = [tag for sentence in tagger_preds for _, tag in sentence]
        preds = [i if i else '-NONE-' for i in preds]
        compare_dict.setdefault(
            name,
            {
                "Accuracy": metrics.accuracy_score(golds, preds),
                "Precision": metrics.precision_score(golds, preds, average='weighted'),
                "Recall": metrics.recall_score(golds, preds, average='weighted'),
                "F1": metrics.f1_score(golds, preds, average='weighted'),
            }
        )
    return pd.DataFrame(compare_dict).T

# helper functions for confusion matrix
def get_tag_list(tagged_sents):
    return [tag for sent in tagged_sents for (word, tag) in sent]

def apply_tagger(tagger, corpus):
    return [tagger.tag(untag(sent)) for sent in corpus]

def get_performance_dataframe(tagger, test_tag_list):
    """Returns DataFrame with metrics for individual tag combinations. For NLTK taggers."""
    truth_sets = defaultdict(set)
    test_sets = defaultdict(set)
    
    for n, (w, label) in enumerate(test_tag_list):
        observed = tagger.tag([w])[0][1]
        truth_sets[label].add(n)
        test_sets[observed].add(n)

    performance_dict = dict()
    for key in test_sets.keys():
        performance_dict.setdefault(
            key,
            {
                'Precision': precision(truth_sets[key], test_sets[key]),
                'Recall': recall(truth_sets[key], test_sets[key]),
                'F1': f_measure(truth_sets[key], test_sets[key])
            }
        )
    df = pd.DataFrame(performance_dict).T
    return df

# spaCy tagger evaluation helper functions
def get_spacy_test_sentences(nltk_test_sentences):
    spacy_test_list = list()
    for s in nltk_test_sentences:
        _word_list = list()
        _tag_list = list()
        for i in s:
            _word_list.append(i[0])
            _tag_list.append(i[1])
        spacy_test_list.append((_word_list, _tag_list))
    return spacy_test_list

def get_spacy_accuracy(spacy_pos_model, spacy_test_list, eval_punct=False):
    scorer = Scorer(eval_punct=eval_punct)
    for tokens, label in spacy_test_list:
        doc = Doc(spacy_pos_model.vocab, words=tokens)
        gold = GoldParse(doc, words=tokens, tags=label)
        processed = spacy_pos_model.tagger(doc)
        scorer.score(processed, gold)
    return scorer.scores['tags_acc']

# textual data munging for LSTM training
def flatten_tagged_sentences(tagged_sentences):
    sentences = list()
    sentence_tags = list()

    for s in tagged_sentences:
        sentence, tags = zip(*s)
        sentences.append(np.array(sentence))
        sentence_tags.append(np.array(tags))
    return sentences, sentence_tags

def get_word2index(train_sentences):
    words = set([])
    for sent in train_sentences:
        for word in sent:
            words.add(word.lower())
    word2index = {word: ind+2 for ind, word in enumerate(list(words))}
    word2index['-OOV-'] = 1
    word2index['-PAD-'] = 0
    return word2index

def get_tag2index(train_tags):
    tags = set([])
    for tag in train_tags:
        for t in tag:
            tags.add(t)
    tag2index = {tag: ind+1 for ind, tag in enumerate(list(tags))}
    tag2index['-PAD-'] = 0
    return tag2index

def sentence2int(sentences, word2index):
    output_sentences = list()
    for s in sentences:
        int_sentences = list()
        for word in s:
            try:
                int_sentences.append(word2index[word.lower()])
            except KeyError:
                int_sentences.append(word2index['-OOV-'])

        output_sentences.append(int_sentences)
    return output_sentences

def tag2int(tags, tag2index):
    int_tags = list()
    for t in tags:
        int_tags.append([tag2index[i] for i in t])
    return int_tags

def one_hot_encoding(seqs, cats):
    categorical_sequences = list()
    for seq in seqs:
        temp_categoricals = list()
        for i in seq:
            temp_categoricals.append(np.zeros(cats))
            temp_categoricals[-1][i] = 1.0
        categorical_sequences.append(temp_categoricals)
    return np.array(categorical_sequences)
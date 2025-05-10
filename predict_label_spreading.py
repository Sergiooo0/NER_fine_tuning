import argparse
import pickle
import sys
from sklearn.feature_extraction import DictVectorizer
from sklearn.semi_supervised import LabelSpreading
from sklearn.preprocessing import LabelEncoder


class IOB:
    def __init__(self, sep=" "):
        self._sep = sep

    def parse_file(self, ifile):
        return [
            self._parse_sentence(raw)
            for raw in self._read_sentences_from_file(ifile)
        ]

    def _parse_sentence(self, raw_sentence):
        return [
            tuple(token.split(self._sep),)
            for token in raw_sentence.strip().split("\n")
        ]

    def _read_sentences_from_file(self, ifile):
        raw_sentence = ""
        try:
            with open(ifile) as fhi:
                for line in fhi:
                    if line == "\n":
                        if raw_sentence:
                            yield raw_sentence
                            raw_sentence = ""
                    else:
                        raw_sentence += line
            if raw_sentence:
                yield raw_sentence
        except IOError:
            print("Unable to read file: " + ifile)
            sys.exit()


class CRFFeatures:
    def word2features(self, sent, i):
        word = sent[i][0]

        features = {
            'bias': 1.0,
            'word.lower()': word.lower(),
            'word[-3:]': word[-3:],
            'word.isupper()': word.isupper(),
            'word.istitle()': word.istitle(),
            'word.isdigit()': word.isdigit(),
        }

        if i > 0:
            word1 = sent[i-1][0]
            features.update({
                '-1:word.lower()': word1.lower(),
                '-1:word.istitle()': word1.istitle(),
                '-1:word.isupper()': word1.isupper(),
            })
        else:
            features['BOS'] = True

        if i < len(sent)-1:
            word1 = sent[i+1][0]
            features.update({
                '+1:word.lower()': word1.lower(),
                '+1:word.istitle()': word1.istitle(),
                '+1:word.isupper()': word1.isupper(),
            })
        else:
            features['EOS'] = True

        return features

    def sent2features(self, sent):
        return [self.word2features(sent, i) for i in range(len(sent))]

    def sent2labels(self, sent):
        return [token[-1] for token in sent]


def parse_args():
    parser = argparse.ArgumentParser(description="CRF prediction with optional label propagation")
    parser.add_argument('-m', '--model', default='crf.es.model', type=str, help='CRF model file')
    parser.add_argument('dataset', type=str, help='input dataset file (IOB2)')
    parser.add_argument('output', type=str, help='output dataset file (IOB2)')
    return parser.parse_args()


def predict_with_label_propagation(args):
    iob = IOB()
    feats = CRFFeatures()

    # Load sentences and CRF model
    sentences = iob.parse_file(args.dataset)
    crf = pickle.load(open(args.model, 'rb'))

    # Prepare features and initial labels
    X_all = []
    y_all = []
    flat_tokens = []

    for sent in sentences:
        X_sent = feats.sent2features(sent)
        y_sent = feats.sent2labels(sent)
        X_all.extend(X_sent)
        y_all.extend(y_sent)
        flat_tokens.extend(sent)

    # Predict with CRF
    y_pred_crf = crf.predict([feats.sent2features(sent) for sent in iob.parse_file(args.dataset)])
    y_pred_flat = [label for sent in y_pred_crf for label in sent]

    # Prepare labels: use CRF predictions where label is missing
    labels = []
    for original, pred in zip(y_all, y_pred_flat):
        if original == "O":  # assume "O" are unlabeled
            labels.append(-1)
        else:
            labels.append(original)

    # Encode labels
    le = LabelEncoder()
    known_labels = [l for l in labels if l != -1]
    le.fit(known_labels)
    y_encoded = [le.transform([l])[0] if l != -1 else -1 for l in labels]

    # Vectorize features
    vec = DictVectorizer(sparse=True)
    X_vect = vec.fit_transform(X_all)

    # Apply label propagation
    label_prop = LabelSpreading(kernel='rbf', alpha=0.8, max_iter=1000)
    label_prop.fit(X_vect, y_encoded)
    y_propagated = label_prop.transduction_
    y_final = le.inverse_transform(y_propagated)

    # Write final output
    with open(args.output, 'w') as f:
        for i, (token, label) in enumerate(zip(flat_tokens, y_final)):
            f.write(f"{token[0]} {label}\n")
            if token[0] in [".", "!", "?"]:  # Heurística para cambio de oración
                f.write("\n")

    print(f"Predictions with label propagation saved to {args.output}")


if __name__ == '__main__':
    args = parse_args()
    predict_with_label_propagation(args)

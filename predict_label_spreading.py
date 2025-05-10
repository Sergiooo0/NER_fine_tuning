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
        sentences = []
        current = []

        with open(ifile, 'r') as f:
            for line in f:
                line = line.strip()
                if line == "":
                    if current:
                        sentences.append(current)
                        current = []
                else:
                    parts = line.split(self._sep)
                    if len(parts) == 2:
                        current.append((parts[0], parts[1]))
                    elif len(parts) == 1:
                        current.append((parts[0], None))
                    else:
                        raise ValueError(f"Línea inválida: {line}")
            if current:
                sentences.append(current)
        return sentences


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
        return [label if label is not None else "O" for _, label in sent]



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

    # Predict with CRF
    print("Predicting with CRF...")
    y_pred_crf = crf.predict([feats.sent2features(sent) for sent in sentences])
    y_pred_flat = [label for sent in y_pred_crf for label in sent]

    # Preparar ventanas móviles con contexto
    def sent2context_features(sent):
        padded = [("BOS",)] * 2 + sent + [("EOS",)] * 2
        features = []
        for i in range(2, len(padded) - 2):
            window = padded[i - 2:i + 3]
            feat = {
                "w-2": window[0][0].lower(),  # Palabra en la posición -2 (pasado)
                "w-1": window[1][0].lower(),  # Palabra en la posición -1
                "w0": window[2][0].lower(),   # Palabra en la posición 0 (actual)
                "w+1": window[3][0].lower(),  # Palabra en la posición +1
                "w+2": window[4][0].lower(),  # Palabra en la posición +2 (futuro)
                "w0.isupper": window[2][0].isupper(),  # Si la palabra actual está en mayúsculas
            }
            features.append(feat)
        return features

    X_all = []
    labels = []
    flat_tokens = []

    print("Preparing context features for label propagation...")
    for sent, pred_labels in zip(sentences, y_pred_crf):
        context_feats = sent2context_features(sent)
        gold_labels = feats.sent2labels(sent)
        for i in range(len(sent)):
            token = sent[i]
            flat_tokens.append(token)

            X_all.append(context_feats[i])

            if gold_labels[i] == "O":
                labels.append(-1)
            else:
                labels.append(gold_labels[i])

    # Encode labels
    le = LabelEncoder()
    known_labels = [l for l in labels if l != -1]
    print(f"Known labels: {set(known_labels)}")
    print("Encoding labels...")
    le.fit(known_labels)
    y_encoded = [le.transform([l])[0] if l != -1 else -1 for l in labels]

    # Vectorize features
    vec = DictVectorizer(sparse=True)
    print("Vectorizing features...")
    X_vect = vec.fit_transform(X_all)

    # Apply label propagation (lighter memory footprint now)
    label_prop = LabelSpreading(kernel='knn', n_neighbors=7, alpha=0.8, max_iter=100, gamma=0.5)
    print("Fitting label propagation model...")
    label_prop.fit(X_vect, y_encoded)
    print("Predicting with label propagation...")
    y_propagated = label_prop.transduction_
    y_final = le.inverse_transform(y_propagated)
    print("Label propagation completed.")

    # Save predictions to output file
    print("Saving predictions to output file...")
    idx = 0
    with open(args.output, 'w') as f:
        for sentence in sentences:
            for token, gold_label in sentence:
                pred_label = y_final[idx]
                f.write(f"{token} {pred_label}\n")
                idx += 1
            f.write("\n")  # línea vacía al final de cada oración


    print(f"Predictions with label propagation (context windows) saved to {args.output}")

if __name__ == '__main__':
    args = parse_args()
    predict_with_label_propagation(args)

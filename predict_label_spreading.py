import argparse
import sys
from sklearn.feature_extraction import DictVectorizer
from sklearn.semi_supervised import LabelSpreading
from sklearn.preprocessing import LabelEncoder

# python predict_label_spreading.py data/ner-es.train.csv  data/ner-es-propagation.train.csv
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


def sent2context_features(sent):
    padded = [("BOS",)] * 2 + sent + [("EOS",)] * 2
    features = []
    for i in range(2, len(padded) - 2):
        window = padded[i - 2:i + 3]
        feat = {
            "w-2": window[0][0].lower(),
            "w-1": window[1][0].lower(),
            "w0": window[2][0].lower(),
            "w+1": window[3][0].lower(),
            "w+2": window[4][0].lower(),
            "w0.isupper": window[2][0].isupper(),
        }
        features.append(feat)
    return features


def parse_args():
    parser = argparse.ArgumentParser(description="Label Propagation on IOB dataset")
    parser.add_argument('dataset', type=str, help='input dataset file (IOB2)')
    parser.add_argument('output', type=str, help='output dataset file (IOB2)')
    return parser.parse_args()


def run_label_propagation(args):
    iob = IOB()
    sentences = iob.parse_file(args.dataset)

    X_all = []
    labels = []
    flat_tokens = []

    print("Preparing context features for label propagation...")
    for sent in sentences:
        context_feats = sent2context_features(sent)
        for i in range(len(sent)):
            token = sent[i]
            flat_tokens.append(token)

            X_all.append(context_feats[i])

            if token[1] is None:
                labels.append(-1)
            else:
                labels.append(token[1])

    le = LabelEncoder()
    known_labels = [l for l in labels if l != -1]
    print(f"Known labels: {set(known_labels)}")
    print("Encoding labels...")
    le.fit(known_labels)
    y_encoded = [le.transform([l])[0] if l != -1 else -1 for l in labels]

    vec = DictVectorizer(sparse=True)
    print("Vectorizing features...")
    X_vect = vec.fit_transform(X_all)

    label_prop = LabelSpreading(kernel='knn', n_neighbors=5, alpha=0.8, max_iter=50)
    print("Fitting label propagation model...")
    label_prop.fit(X_vect, y_encoded)
    y_propagated = label_prop.transduction_

    # Construimos etiquetas finales, preservando las originales
    y_final = []
    for orig_label, propagated in zip(labels, y_propagated):
        if orig_label != -1:
            y_final.append(orig_label)
        else:
            y_final.append(le.inverse_transform([propagated])[0])

    print("Label propagation completed.")

    print("Saving predictions to output file...")
    idx = 0
    with open(args.output, 'w') as f:
        for sentence in sentences:
            for token, _ in sentence:
                pred_label = y_final[idx]
                f.write(f"{token} {pred_label}\n")
                idx += 1
            f.write("\n")

    print(f"Predictions saved to {args.output}")


if __name__ == '__main__':
    args = parse_args()
    run_label_propagation(args)

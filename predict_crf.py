import argparse
import pickle
import sys


"""
Para ejecutar el script:
python predict_crf.py data/ner-es.train.csv data/ner-es-complete.train.csv
Se puede usar otro modelo con el argumento -m, por ejemplo:
python predict_crf.py -m crf.es.model data/ner-es.train.csv data/ner-es-complete.train.csv
"""
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
                        if raw_sentence == "":
                            continue
                        yield raw_sentence
                        raw_sentence = ""
                        continue

                    if line:
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
            'word.len()': len(word),
        }

        # Sufijos y prefijos
        features.update({
            'word[-2:]': word[-2:],
            'word[-1:]': word[-1:],
            'word[:2]': word[:2],
            'word[:3]': word[:3],
        })

        # Características de la palabra anterior y posterior
        if i > 0:
            word1 = sent[i-1][0]
            features.update({
                '-1:word.lower()': word1.lower(),
                '-1:word.istitle()': word1.istitle(),
                '-1:word.isupper()': word1.isupper(),
            })
        else:
            features['BOS'] = True  # Beginning of sentence

        if i < len(sent)-1:
            word1 = sent[i+1][0]
            features.update({
                '+1:word.lower()': word1.lower(),
                '+1:word.istitle()': word1.istitle(),
                '+1:word.isupper()': word1.isupper(),
            })
        else:
            features['EOS'] = True  # End of sentence

        return features


    def sent2features(self, sent):
        return [self.word2features(sent, i) for i in range(len(sent))]

    def sent2labels(self, sent):
        return [token[-1] for token in sent]


def parse_args():
    description = ""

    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '-m',
        '--model',
        default='crf.es.model',
        type=str,
        metavar='FILE',
        help='model file',
    )
    parser.add_argument(
        'dataset',
        metavar='input file',
        type=str,
        help='dataset file (IOB2)'
    )
    parser.add_argument(
        'output',
        metavar='output file',
        type=str,
        help='output dataset file (IOB2)'
    )
    return parser.parse_args()


def predict(args):
    iob = IOB()
    crf = pickle.load(open(args.model, 'rb'))
    feats = CRFFeatures()

    sentences = [
        [tuple(token) for token in sent]
        for sent in iob.parse_file(args.dataset)
    ]

    X = [feats.sent2features(s) for s in sentences]
    y_pred = crf.predict(X)

    # Guardar el resultado en el archivo de salida:
    with open(args.output, 'w') as f:
        idx = 0  
        for i, sentence in enumerate(sentences):
            for j, token in enumerate(sentence):
                word = token[0]  
                true_label = token[1] if len(token) > 1 else None  # Asigna None si no hay etiqueta  

                # Escribe la predicción solo si la palabra no tiene etiqueta
                if true_label == None:  # Solo predice en los casos donde la etiqueta es None
                    f.write("{} {}\n".format(word, y_pred[i][j]))
                else:
                    f.write("{} {}\n".format(word, true_label))  # Usa la etiqueta verdadera si existe
            f.write("\n")  # Escribe una nueva línea después de cada oración

    print("Predictions saved to {}".format(args.output))



if __name__ == '__main__':
    args = parse_args()
    predict(args)
import argparse
import json
import sys


class IOB:
    def __init__(self, sep=" "):
        self._sep = sep

    def parse_file(self, ifile):
        return [
            self._parse_sentence(raw)
            for raw in self._read_sentences_from_file(ifile)
        ]

    def _parse_sentence(self, raw_sentence):
        sentence = []
        for line in raw_sentence.strip().split("\n"):
            parts = line.strip().split(self._sep)
            if len(parts) != 2:
                print(f"Advertencia: liña mal formada omitida -> '{line}'", file=sys.stderr)
                continue
            token, label = parts
            sentence.append((token, label))
        return sentence

    def _read_sentences_from_file(self, ifile):
        raw_sentence = ""
        try:
            with open(ifile, encoding="utf-8") as fhi:
                for line in fhi:
                    if line.strip() == "":
                        if raw_sentence.strip():
                            yield raw_sentence.strip()
                            raw_sentence = ""
                    else:
                        raw_sentence += line
                if raw_sentence.strip():
                    yield raw_sentence.strip()
        except IOError:
            print(f"Non se puido ler o ficheiro: {ifile}", file=sys.stderr)
            sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(description="Convert IOB format to JSONL.")
    parser.add_argument("iobfile", help="Ficheiro de entrada no formato IOB")
    parser.add_argument("jsonfile", help="Ficheiro de saída en formato JSONL")
    return parser.parse_args()


def convert_to_json(ifile, ofile):
    iob = IOB()
    sentences = iob.parse_file(ifile)

    with open(ofile, "w", encoding="utf-8") as f:
        for sentence in sentences:
            if sentence:  
                json_obj = {
                    "tokens": [token for token, _ in sentence],
                    "labels": [label for _, label in sentence],
                }
                f.write(json.dumps(json_obj, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    args = parse_args()
    convert_to_json(args.iobfile, args.jsonfile)

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
        return [
            tuple(token.split(self._sep))
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


def parse_args():
    parser = argparse.ArgumentParser(description="Convert IOB format to JSON.")
    parser.add_argument(
        "iobfile",
        help="Input file in IOB format"
    )
    parser.add_argument(
        "jsonfile",
        help="Output file in JSON format"
    )
    return parser.parse_args()


def convert_to_json(ifile, ofile):
    iob = IOB()
    sentences = iob.parse_file(ifile)

    jsonl = [
        {
            "tokens": [token[0] for token in sentence],
            "labels": [token[1] for token in sentence],
        }
        for sentence in sentences
    ]

    with open(ofile, "w") as f:
        for sentence in jsonl:
            f.write(json.dumps(sentence, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    args = parse_args()
    convert_to_json(args.iobfile, args.jsonfile)

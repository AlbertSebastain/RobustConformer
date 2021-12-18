"""Build vocabulary from manifest files.

Each item in vocabulary file is a character.
"""
import sys
import argparse
import functools
import codecs
import json
from collections import Counter
import os.path


def count_manifest(manifest_path):
    collection = []
    with open(manifest_path) as f:
        for line in f:
            line_splits = line.strip().split()
            utt_id = line_splits[0]
            # transcript = "".join(line_splits[1:])
            transcript = line_splits[1:]
            for word in transcript:
                collection.append(word)
    return Counter(collection)
    # for word in transcript:
    #     counter.update(word)


def main():
    # text = sys.argv[1]
    # count_threshold = int(sys.argv[2])
    # vocab_path = sys.argv[3]

    text = "/usr/home/shi/projects/data_aishell/data/train/text"
    count_threshold = 1
    vocab_path = "data/build_word_result.txt"

    # counter = Counter()
    counter = count_manifest(text)

    count_sorted = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    print(len(count_sorted))
    num = 1
    with open(vocab_path, "w") as fout:
        fout.write("<unk> 1" + "\n")
        for char, count in count_sorted:
            if count < count_threshold:
                break
            num += 1
            fout.write(char + " " + str(num) + "\n")
    print(num)


if __name__ == "__main__":
    main()

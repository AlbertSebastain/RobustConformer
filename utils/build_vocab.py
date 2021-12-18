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
def count_manifest_word(manifest_path):
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
def count_manifest(counter, manifest_path):
    with open(manifest_path) as f:
        for line in f:
            line_splits = line.strip().split()
            utt_id = line_splits[0]
            transcript = ''.join(line_splits[1:])
            for char in transcript:
                counter.update(char)
def count_manifest_word_vocab(manifest_path):
    collections = []
    with open(manifest_path) as f:

        for line in f:

            line_splits = line.strip().split()
            utt_id = line_splits[0]
            transcript = ''.join(line_splits[1:])
            for char in transcript:
                collections.append(char)
                #counter.update(char)
            for word in line_splits[1:]:
                #counter.update(word)
                collections.append(word)
    return Counter(collections)
def main():
    text = sys.argv[1]
    count_threshold = int(sys.argv[2])
    vocab_path = sys.argv[3]
    type_dic = sys.argv[4]
    #text = "/usr/home/shi/projects/data_aishell/data/train/text"
    #count_threshold = 1
    #vocab_path = "train_units.txt"
    #type_dic = "both"
    #except expression as identifier:
        #pass
    if type_dic == "vocab":
        
        counter = Counter()
        count_manifest(counter, text)
        ount_sorted = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        print (len(ount_sorted))
        num = 1
        with open(vocab_path, 'w') as fout:
            fout.write('<unk> 1' + '\n')
            for char, count in ount_sorted:
                if count < count_threshold: break
                num += 1
                fout.write(char + ' ' + str(num) + '\n')
        print ("num=",num)
    elif type_dic == "word":
        counter = count_manifest_word(text)
        ount_sorted = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        print (len(ount_sorted))
        num = 1
        with open(vocab_path, 'w') as fout:
            fout.write('<unk> 1' + '\n')
            for char, count in ount_sorted:
                if count < count_threshold: break
                num += 1
                fout.write(char + ' ' + str(num) + '\n')
    else:
        #counter = Counter()
        #count_manifest_word_vocab(counter,text)
        #print(ount_sorted['市'])
        #print(ount_sorted('楼市'))
        counter = count_manifest_word_vocab(text)
        ount_sorted = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        print (len(ount_sorted))
        num = 1
        with open(vocab_path, 'w') as fout:
            fout.write('<unk> 1' + '\n')
            for char, count in ount_sorted:

                if count < count_threshold: break
                num += 1
                fout.write(char + ' ' + str(num) + '\n')
    


if __name__ == '__main__':
    main()


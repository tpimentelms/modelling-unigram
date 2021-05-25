import sys
import logging
import argparse
import string
import numpy as np
from tqdm import tqdm

sys.path.append('./src/')
from h01_data.language_characters import get_character_set
from util import util


def get_args():
    parser = argparse.ArgumentParser(description='DataFilter')
    parser.add_argument("--input-file", type=str,
                        help="The file in which raw tokenized data is")
    parser.add_argument("--output-file", type=str,
                        help="The file in which filtered data should be saved")
    parser.add_argument("--language", type=str,
                        help="The language the data is in")
    return parser.parse_args()


def count_lines(fname):
    count = 0
    with open(fname, 'r') as f:
        for _ in f:
            count += 1
    return count


def is_allowed(word, char_set):
    return all([char in char_set for char in word.lower()])


def filter_line(line, language):
    # remove punctuation
    line = line.translate(str.maketrans('', '', string.punctuation))
    sentence = [word.lower() for word in list(filter(None, line.strip().split(' ')))]

    character_set = get_character_set(language)
    line_new = ' '.join([word for word in sentence if is_allowed(word, character_set)])
    return line_new


def append_to_file(tgt_fname, line):
    if not line:
        return

    with open(tgt_fname, 'a') as f:
        f.write(line + '\n')


def filter_data(src_fname, tgt_fname, language):
    n_lines = count_lines(src_fname)

    with open(src_fname, 'r') as f:
        for line in tqdm(f, total=n_lines, desc='Processing wiki data'):
            line_new = filter_line(line, language)
            append_to_file(tgt_fname, line_new)


def process(src_fname, tgt_fname, language):
    assert util.is_file(src_fname), 'Input file should exist'
    assert not util.is_file(tgt_fname), 'Output file should not exist'

    filter_data(src_fname, tgt_fname, language)


def main():
    args = get_args()
    logging.info(args)

    process(args.input_file, args.output_file, args.language)


if __name__ == '__main__':
    main()

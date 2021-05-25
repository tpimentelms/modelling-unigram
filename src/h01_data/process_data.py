import sys
import logging
import argparse
import string
import numpy as np
from tqdm import tqdm

sys.path.append('./src/')
# from h01_data.alphabet import Alphabet
from h01_data.language_characters import get_character_set
# from util.argparser import get_argparser, parse_args, add_data_args
from util import util


def get_args():
    parser = argparse.ArgumentParser(description='DataFilter')
    parser.add_argument(
        "--input-file", type=str,
        help="The file in which raw tokenized data is")
    parser.add_argument(
        "--output-file", type=str,
        help="The file in which filtered data should be saved")
    parser.add_argument(
        "--language", type=str,
        help="The language the data is in")
    return parser.parse_args()


def count_lines(fname):
    count = 0
    with open(fname, 'r') as f:
        for _ in f:
            count += 1
    return count


# def get_fold_splits(n_sentences, n_folds, max_sentences=None):
#     splits = np.arange(n_sentences)
#     np.random.shuffle(splits)
#     if max_sentences is not None:
#         splits = splits[:max_sentences]
#     splits = np.array_split(splits, n_folds)
#     splits = {x: i for i, fold in enumerate(splits) for x in fold}
#     return splits


def is_allowed(word, char_set):
    return all([char in char_set for char in word.lower()])


def filter_line(line, language):
    character_set = get_character_set(language)
    # remove punctuation
    line = line.translate(str.maketrans('', '', string.punctuation))
    sentence = [word.lower() for word in list(filter(None, line.strip().split(' ')))]
    # only accept words without extra symbols
    line_new = ' '.join([word for word in sentence if is_allowed(word, character_set)])

    return line_new
    # if not keep:
    #     return
    # sentence_list.append(sentence)
    # for word in sentence:
    #     # exclude words that contain non-letters
    #     word = word.lower()
    #     alphabet.add_word(word)

    #     if word in word_info:
    #         word_info[word]['count'] += 1
    #     else:
    #         word_info[word] = {
    #             'count': 1,
    #             'idx': alphabet.word2idx(word)
    #         }


def append_to_file(tgt_fname, line):
    if not line:
        return

    with open(tgt_fname, 'a') as f:
        f.write(line + '\n')


# def process_data(src_fname, n_folds, splits, alphabet, language):
def filter_data(src_fname, tgt_fname, language):
    # word_folds = [{} for _ in range(n_folds)]
    # sentence_folds = [[] for _ in range(n_folds)]
    n_lines = count_lines(src_fname)

    with open(src_fname, 'r') as f:
        for line in tqdm(f, total=n_lines, desc='Processing wiki data'):
            line_new = filter_line(line, language)
            append_to_file(tgt_fname, line_new)

    # return word_folds, sentence_folds


def count_tokens(folds):
    return [sum([x['count'] for x in word_info.values()]) for word_info in folds]


def count_types(folds):
    return [len(word_info) for word_info in folds]


def process(src_fname, tgt_fname, language):
    # spacy_tokenizer = load_spacy(spacy_option)
    # splits = get_fold_splits(n_lines, n_folds, max_sentences=max_sentences)
    # alphabet = Alphabet()

    assert util.is_file(src_fname), 'Input file should exist'
    assert not util.is_file(tgt_fname), 'Output file should not exist'

    filter_data(src_fname, tgt_fname, language)
    # n_tokens = count_tokens(word_folds)
    # n_types = count_types(word_folds)
    # util.write_data(tgt_fname, (word_folds, sentence_folds, alphabet, n_tokens))

    # print('# unique chars:', len(alphabet))
    # print('# tokens per fold:', n_tokens)
    # print('# types per fold:', n_types)


def main():
    args = get_args()
    logging.info(args)

    process(args.input_file, args.output_file, args.language)


if __name__ == '__main__':
    main()

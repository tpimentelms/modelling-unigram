import os
import sys
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append('./src/')
from h02_learn.dataset import load_data
from h02_learn.dataset.tokens import TokenDataset
from util.argparser import get_argparser, parse_args, add_data_args

def calculate_word_freqs(dataset):
    fold_freqs = dataset.word_freqs
    word_freqs = defaultdict(int)
    for w, f in fold_freqs:
        word_freqs[w] += f
    return word_freqs    

def get_sorted_freqs(dataset):
    word_freqs = calculate_word_freqs(dataset)
    sorted_freqs = sorted(word_freqs.items(), key=lambda x: -x[1])    
    return sorted_freqs

def get_ranks(sorted_freqs):
    ranks = []
    freqs = []
    for i, word_freq in enumerate(sorted_freqs):
        ranks.append(i)
        freqs.append(word_freq[1])
    return ranks, freqs

def get_word_ranks(dataset):
    sorted_freqs = get_sorted_freqs(dataset)
    ranks = {}
    for i, word_freq in enumerate(sorted_freqs):
        freqs[word_freq[0]] = i
    return ranks

def get_ranks_and_freqs(dataset):
    sorted_freqs = get_sorted_freqs(dataset)
    ranks, freqs = get_ranks(sorted_freqs)
    return ranks, freqs

def get_args():
    argparser = get_argparser()
    argparser.add_argument('--data-folder', type=str, required=True)
    args = parse_args(argparser)
    return args

def main():
    args = get_args()
    langs = ['fi', 'yo', 'he', 'id', 'en', 'ta', 'tr']
    for lang in langs:
        print('Getting data for', lang)
        data_path = os.path.join(args.data_folder, lang, 'processed.pckl')
        data = load_data(data_path)
        dataset = TokenDataset(data, list(range(10)))
        print('Dataset size for', lang, ':', len(dataset))
        ranks, freqs = get_ranks_and_freqs(dataset)
        # plot in logscale
        plt.plot(ranks, freqs, label=lang)
        plt.yscale('log')
        plt.xscale('log')
    plt.title('Zip\'s law in the Wikipedia corpus')
    plt.xlabel('Word rank')
    plt.ylabel('Word frequency')
    plt.legend(loc="upper right", prop={'size': 13})
    plt.show()

if __name__ == '__main__':
    main()

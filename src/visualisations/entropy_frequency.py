import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append('./src/')
from h02_learn.dataset import get_data_loaders_with_folds
from h02_learn.dataset.tokens import TokenDataset
from h02_learn.train_generator import load_generator
from h02_learn.model.adaptor import Adaptor
from h03_eval.eval_generator import load_model
from util.argparser import get_argparser, parse_args
from util import util
from .zipfs_law import calculate_word_freqs, get_word_ranks

def get_args():
    argparser = get_argparser()
    argparser.add_argument('--max-train-tokens', type=int, required=True)
    argparser.add_argument('--data-language-dir', type=str, required=True)
    argparser.add_argument('--checkpoint-language-dir', type=str, required=True)
    argparser.add_argument('--two-stage-state-folder', type=str, required=True)
    argparser.add_argument('--results-file', type=str, required=True)
    args = parse_args(argparser)
    return args


def get_model(model_name, args):
    model_path = os.path.join(args.checkpoint_language_dir, model_name + '_' + args.max_train_tokens)
    model = load_model(model_path)
    return model


def get_lm_loss(model_name, x, y, args):
    model = get_model(model_name, args)
    y_hat = model(x)
    loss = model.get_loss(y_hat, y).sum(-1)
    return loss.item()


def get_two_stage_loss(adaptor, generator, x, y, word):
    generator_logprob = generator.get_word_log_probability(x, y)
    two_stage_logprob = adaptor.get_token_logprobability(generator_logprob, word)
    return -two_stage_logprob


def main():
    args = get_args()
    folds = [list(range(8)), [8], [9]]
    
    data_file = os.path.join(args.data_language_dir, 'processed.pckl')

    _, _, type_testloader, _ = get_data_loaders_with_folds(
        'types', data_file, folds,
        batch_size=1, test=True)
    _, _, token_testloader, _ = get_data_loaders_with_folds(
        'tokens', data_file, folds,
        batch_size=1, max_train_tokens=args.max_train_tokens, test=True)

    generator = load_generator(args.two_stage_state_folder)
    generator.eval()
    adaptor = Adaptor.load(args.two_stage_state_folder)

    word_freqs = calculate_word_freqs(token_testloader.dataset)
    word_ranks = get_word_ranks(token_testloader.dataset)

    results = ['type_loss', 'token_loss', 'two_stage_entr', 'freq', 'rank']
    for x, y, _, _, word in type_testloader:
        type_loss = get_lm_loss('types', x, y, args)
        token_loss = get_lm_loss('tokens', x, y, args)
        two_stage_loss = get_two_stage_loss(adaptor, generator, x, y, word)
        freq = word_freqs[word]
        rank = word_ranks[word]
        results += [[type_loss, token_loss, two_stage_loss, freq, rank]]

    util.overwrite_csv(args.results_file, results)

if __name__ == '__main__':
    main()

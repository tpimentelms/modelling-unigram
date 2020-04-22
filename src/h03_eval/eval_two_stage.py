import os
import sys
import torch

sys.path.append('./src/')
from h02_learn.dataset import get_data_loaders_with_folds
from h02_learn.train import load_generator
from h02_learn.adaptor import Adaptor
from util import util
from util import argparser

def get_args():
    # Save
    argparser.add_argument('--adaptor-results-file', type=str, required=True)
    # adaptor
    argparser.add_argument('--two-stage-state-folder', type=str, required=True)
    args = argparser.parse_args()
    return args

def evaluate_adaptor(dataloader, generator, adaptor):
    print('Evaluating adaptor with a dataset of size', len(dataloader.dataset))
    generator.eval()
    dataloader.dataset.eval()
    with torch.no_grad():
        cross_entropy = adaptor.calculate_cross_entropy(dataloader, generator)
    generator.train()
    dataloader.dataset.train()
    return cross_entropy

def save_pitman_yor_results(model, alpha, beta, train_loss, dev_loss, test_loss, test_size, results_fname):
    print('Saving to', results_fname)
    results = []
    file_size = os.path.getsize(results_fname) if os.path.exists(results_fname) else 0
    if file_size == 0:
        results = [['alphabet_size', 'embedding_size', 'hidden_size', 'nlayers',
                    'dropout_p', 'alpha', 'beta', 'train_loss', 'dev_loss', 'test_loss', 'test_size']]
    results += [[model.alphabet_size, model.embedding_size, model.hidden_size, model.nlayers,\
                model.dropout_p, alpha, beta, train_loss, dev_loss, test_loss, test_size]]
    util.write_csv(results_fname, results)

def main():
    args = get_args()
    folds = [list(range(8)), [8], [9]]

    trainloader, devloader, testloader, alphabet = \
        get_data_loaders_with_folds(args.dataset, args.data_file, folds,\
                                        args.batch_size, test=True)
    print('Train size: %d Dev size: %d Test size: %d' %
          (len(trainloader.dataset), len(devloader.dataset), len(testloader.dataset)))

    generator = load_generator(alphabet, args.two_stage_state_folder)
    generator.eval()
    adaptor = Adaptor.load(args.two_stage_state_folder)

    train_loss = evaluate_adaptor(trainloader, generator, adaptor)
    dev_loss = evaluate_adaptor(devloader, generator, adaptor)
    test_loss = evaluate_adaptor(testloader, generator, adaptor)

    print('Adaptor Training loss: %.4f Dev loss: %.4f Test loss: %.4f' %
          (train_loss, dev_loss, test_loss))

    save_pitman_yor_results(generator, adaptor.state['alpha'], adaptor.state['alpha'], train_loss, dev_loss, test_loss,\
                                len(testloader.dataset), args.adaptor_results_file)


if __name__ == '__main__':
    main()

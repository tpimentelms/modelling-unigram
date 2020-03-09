import os
from collections import defaultdict
import torch
import numpy as np
from tqdm import tqdm
from util.util import hacked_exp, write_data, read_data

class Adaptor:
    def __init__(self, alpha, beta, alphabet, dataloader, state_filename='saved_models/saved_adaptor_state'):
        # initialise mapping from table index to n.o. customers (c)
        # int --> int
        self.customers_per_table = defaultdict(int)
        # initialise mapping from table indices to labels (t)
        # int --> list(int)
        self.tables_with_word_label = defaultdict(set)
        # initialise mapping from customer id to table id (z)
        # int --> int
        self.table_assignments = {}
        # this index doesn't have to be "accurate"
        # there may be gaps in the indices as some tables are removed
        # but we just want to make sure that every table index is unique
        self.max_table_index = -1
        # this is marked as the function K in the original paper
        self.total_tables = 0
        self.alpha = torch.Tensor([alpha])
        self.beta = torch.Tensor([beta])
        self.token_dataloader = dataloader
        self.alphabet = alphabet
        curr_dir = os.path.dirname(os.path.realpath(__file__))
        self.saved_state_file = os.path.join(curr_dir, state_filename)
        print('Token data length', len(self.token_dataloader))

    def _sample_new_table_assignment(self, table_probs):
        ids = [idd for prob, idd in table_probs]
        probs = [prob for prob, idd in table_probs]
        table_index = np.random.choice(ids, 1, p=probs)[0]
        if table_index < 0:
            # choose new table index
            # increment counter for total amount of tables
            self.total_tables += 1
            # increment table id counter
            self.max_table_index += 1
            return self.max_table_index
        return table_index

    def calculate_cross_entropy(self, dataloader, generator):
        entropy = 0
        total_tokens = 0
        for x, y, weights in tqdm(dataloader, total=len(dataloader), \
                                    desc='Calculating adaptor cross entropy', mininterval=.2):
            generator_logprobs = generator.get_word_log_probability(x, y)
            for i, log_prob in enumerate(generator_logprobs):
                # do not use the start of word index
                token_indices = x[i][1:]
                token = ''.join(self.alphabet.idx2word(token_indices))
                word_logprob = self.get_token_probability(log_prob, token)
                entropy += -word_logprob * weights[i]
                total_tokens += weights[i]
                #print("word entropy", -word_logprob * weights[i])
                #print("intermediate entropy", entropy / total_tokens)
        return (entropy / total_tokens).item()

    def get_token_probability(self, generator_logprob, token):
        # TODO change alpha and beta after retraining
        # TODO: replace dataset length after retraining
        #print('a', self.alpha, 'b', self.beta)
        i = len(self.state['dataset_length'])
        tables_with_word_label = self.state['tables_with_word_label'][token]
        customers_in_tables_with_label = self.state['customers_in_tables_with_label'][token]
        if len(tables_with_word_label) == 0 and customers_in_tables_with_label == 0:
            # this takes care of rare words not encountered in training
            # their probabilities are too small to take away from log space
            adaptor_state = self.state['total_tables']*self.alpha + self.beta
            return torch.log(adaptor_state) + generator_logprob - torch.log(i+self.beta)
        generator_prob = torch.exp(generator_logprob)
        state1 = customers_in_tables_with_label - len(tables_with_word_label)*self.alpha
        state2 = self.state['total_tables']*self.alpha + self.beta
        res = torch.log(state1 + state2*generator_prob)-torch.log(i+self.beta)
        return res

    def count_customers_in_tables_with_label(self):
        c_in_tables_with_label = defaultdict(int)
        for x, _, _ in self.token_dataloader:
            for word_indices in x:
                word = ''.join(self.alphabet.idx2word(word_indices[1:]))
                c_in_tables_with_label[word] = sum([self.customers_per_table[table_id] \
                                        for table_id in self.tables_with_word_label[word]])
        return c_in_tables_with_label

    def save_fitted_adaptor(self):
        self.state = {}
        self.state['tables_with_word_label'] = self.tables_with_word_label
        self.state['total_tables'] = self.total_tables
        self.state['alpha'] = self.alpha
        self.state['beta'] = self.beta
        self.state['dataset_length'] = len(self.token_dataloader.dataset)
        customers_in_tables_with_label = self.count_customers_in_tables_with_label()
        self.state['customers_in_tables_with_label'] = customers_in_tables_with_label
        write_data(self.saved_state_file, self.state)

    def load_fitted_adaptor(self):
        print('Loading fitted adaptor from', self.saved_state_file)
        self.state = read_data(self.saved_state_file)

    @staticmethod
    def _normalise_table_probabilities(table_logprobs):
        exp_probs = hacked_exp([prob for prob, idd in table_logprobs])
        normaliser = sum(exp_probs)
        table_probs = [(prob/normaliser, table_logprobs[i][1]) \
                            for i, prob in enumerate(exp_probs)]
        return table_probs

    def _calculate_table_logprobs(self, token, token_logprob):
        table_logprobs = []
        # calculate probability of assigning to old table
        for table_id in self.tables_with_word_label[token]:
            table_prob = torch.log(self.customers_per_table[table_id] - self.alpha)
            table_logprobs.append((table_prob.item(), table_id))
        # calculate probability of assigning to new table
        new_table_prob = torch.log(torch.Tensor([self.total_tables*self.alpha + self.beta])) + \
                            token_logprob
        table_logprobs.append((new_table_prob.item(), -1))
        return table_logprobs

    def fit(self, generator, iterations):
        for iteration in range(iterations):
            for x, y, token_ids in tqdm(self.token_dataloader, total=len(self.token_dataloader), \
                                        desc='Fitting adaptor', mininterval=.2):
                tokens_logprobs = generator.get_word_log_probability(x, y)
                # iterate through tokens in batch:
                for i, token_logprob in enumerate(tokens_logprobs):
                    token_id = token_ids[i].item()
                    token_indices = x[i][1:]
                    token = ''.join(self.alphabet.idx2word(token_indices))
                    if token_id in self.table_assignments:
                        token_table_id = self.table_assignments[token_id]
                        # remove customer from table
                        self.customers_per_table[token_table_id] -= 1
                        # if table is empty then don't associate with word anymore
                        if self.customers_per_table[token_table_id] == 0:
                            self.tables_with_word_label[token].remove(token_table_id)
                            self.total_tables -= 1
                    table_logprobs = self._calculate_table_logprobs(token, token_logprob)
                    # normalise to probabilities before sampling
                    table_probs = self._normalise_table_probabilities(table_logprobs)
                    assigned_table_id = self._sample_new_table_assignment(table_probs)
                    # put customer to new table
                    self.customers_per_table[assigned_table_id] += 1
                    # store info about amount of labels
                    self.tables_with_word_label[token].add(assigned_table_id)
                    self.table_assignments[token_id] = assigned_table_id
            print('Saving adaptor state to', self.saved_state_file)
            self.save_fitted_adaptor()
        print('Done fitting the adaptor')

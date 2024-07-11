import argparse
import os

import numpy as np
import torch as t
from torch.optim import Adam

from utils.batch_loader import BatchLoader
from utils.parameters import Parameters
from model.rvae import RVAE

if __name__ == "__main__":

    if not os.path.exists('data/word_embeddings.npy'):
        raise FileNotFoundError("word embeddings file was't found")

    parser = argparse.ArgumentParser(description='RVAE')
    parser.add_argument('--num-iterations', type=int, default=120000, metavar='NI',
                        help='num iterations (default: 120000)')
    parser.add_argument('--batch-size', type=int, default=32, metavar='BS',
                        help='batch size (default: 32)')
    parser.add_argument('--use-cuda', type=bool, default=True, metavar='CUDA',
                        help='use cuda (default: True)')
    parser.add_argument('--learning-rate', type=float, default=0.00005, metavar='LR',
                        help='learning rate (default: 0.00005)')
    parser.add_argument('--dropout', type=float, default=0.3, metavar='DR',
                        help='dropout (default: 0.3)')
    parser.add_argument('--use-trained', type=bool, default=False, metavar='UT',
                        help='load pretrained model (default: False)')
    parser.add_argument('--ce-result', default='', metavar='CE',
                        help='ce result path (default: '')')
    parser.add_argument('--kld-result', default='', metavar='KLD',
                        help='ce result path (default: '')')

    args = parser.parse_args()


    path=''
    
    ''' =================== Creating batch_loader for encoder-1 =========================================
    '''
    data_files = [path + 'data/train.txt',
                       path + 'data/test.txt']

    idx_files = [path + 'data/words_vocab.pkl',
                      path + 'data/characters_vocab.pkl']

    tensor_files = [[path + 'data/train_word_tensor.npy',
                          path + 'data/valid_word_tensor.npy'],
                         [path + 'data/train_character_tensor.npy',
                          path + 'data/valid_character_tensor.npy']]

    batch_loader = BatchLoader(data_files, idx_files, tensor_files, path)
    parameters = Parameters(batch_loader.max_word_len,
                            batch_loader.max_seq_len,
                            batch_loader.words_vocab_size,
                            batch_loader.chars_vocab_size)


    ''' =================== Doing the same for encoder-2 ===============================================
    '''
    data_files = [path + 'data/super/train_2.txt',
                       path + 'data/super/test_2.txt']

    idx_files = [path + 'data/super/words_vocab_2.pkl',
                      path + 'data/super/characters_vocab_2.pkl']

    tensor_files = [[path + 'data/super/train_word_tensor_2.npy',
                          path + 'data/super/valid_word_tensor_2.npy'],
                         [path + 'data/super/train_character_tensor_2.npy',
                          path + 'data/super/valid_character_tensor_2.npy']]
    batch_loader_2 = BatchLoader(data_files, idx_files, tensor_files, path)
    parameters_2 = Parameters(batch_loader_2.max_word_len,
                            batch_loader_2.max_seq_len,
                            batch_loader_2.words_vocab_size,
                            batch_loader_2.chars_vocab_size)
    '''=================================================================================================
    '''


    rvae = RVAE(parameters,parameters_2,parameters_2)
    print(rvae.learnable_parameters())

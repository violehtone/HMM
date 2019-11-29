#!/usr/bin/python3

"""
DESCRIPTION:
    Template code for the SECOND Advanced Question of the Hidden Markov Models
    assignment in the Algorithms in Sequence Analysis course at the VU.

INSTRUCTIONS:
    Complete the code (compatible with Python 3!) upload to CodeGrade via
    corresponding Canvas assignment. Note this script is graded automatically,
    if and only if your "hmm.py" script succesfully implements Baum-Welch
    training!

AUTHOR:
    <your name and student number here>
"""

import os.path as op

from os import makedirs
from argparse import ArgumentParser, RawTextHelpFormatter
from hmm_utility import load_fasta, load_tsv, serialize
from hmm import viterbi



def parse_args():
    "Parses inputs from commandline and returns them as a Namespace object."

    parser = ArgumentParser(prog = 'python3 viterbi_training.py',
        formatter_class = RawTextHelpFormatter, description =
        '  Perform Viterbi training, given a set of sequences with A and E priors.\n\n'
        '  Example syntax:\n'
        '    python3 hmm.py seq.fasta A.tsv E.tsv -i 100 -o /viterbi_outputs'
        '    python3 hmm.py baumwelch in.fa priorA priorE -o ./outputs -i 1')

    # Positionals
    parser.add_argument('fasta', help='path to a FASTA formatted input file')
    parser.add_argument('transition', help='path to a TSV formatted transition matrix')
    parser.add_argument('emission', help='path to a TSV formatted emission matrix')

    # Optionals
    parser.add_argument('-o', dest='out_dir',
        help='path to a directory where output files are saved\n'
             '  (directory will be made if it does not exist)')
    parser.add_argument('-i', dest='max_iter', type=int, default=20,
        help='maximum number of iterations (default: 20 )')

    return parser.parse_args()



def train_viterbi(X,A,E):
    #####################
    # START CODING HERE #
    #####################
    # Initialize your posterior matrices
    new_A = {}    
    # for k in A: ...

    new_E = {}
    # for k in E: ...
    

    # Get the state path of every sequence in X,
    # using the viterbi() function imported from hmm.py
    for seq,label in X:
        # ...
        pass

        # Count the transitions and emissions for every state


    # Normalize your row sums


    #####################
    #  END CODING HERE  #
    #####################
    
    return new_A, new_E


def main(args = False):
    "Perform Viterbi training, given a set of sequences with A and E priors."
    
    # Process arguments and load specified files
    if not args: args = parse_args()

    set_X, labels = load_fasta(args.fasta) # List of sequences, list of labels
    A = load_tsv(args.transition) # Nested Q -> Q dictionary
    E = load_tsv(args.emission)   # Nested Q -> S dictionary
    
    i_max = args.max_iter
    
    #####################
    # START CODING HERE #
    #####################
    # Iterate until you've reached i_max or until your parameters have converged!
    # Note Viterbi converges discretely (unlike Baum-Welch), so you don't need to
    # track your Sum Log-Likelihood to decide this.


    #####################
    #  END CODING HERE  #
    #####################

    
    if args.out_dir:
        makedirs(args.out_dir, exist_ok=True) # Make sure the output directory exists.
        A_path = op.join(args.out_dir,'viterbi_posterior_A')
        with open(A_path,'w') as f: f.write(serialize(A))
        E_path = op.join(args.out_dir,'viterbi_posterior_E')
        with open(E_path,'w') as f: f.write(serialize(E))        



if __name__ == "__main__":
    main()
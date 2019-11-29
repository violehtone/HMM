#!/usr/bin/python3

"""
DESCRIPTION:
    Template code for the FIRST Advanced Question of the Hidden Markov Models
    assignment in the Algorithms in Sequence Analysis course at the VU.

INSTRUCTIONS:
    Complete the code (compatible with Python 3!) upload to CodeGrade via
    corresponding Canvas assignment. Note this script will be graded manually,
    if and only if your "hmm.py" script succesfully implements Baum-Welch
    training! Continuous Feedback will not be available for this script.

AUTHOR:
    <your name and student number here>
"""

from argparse import ArgumentParser, RawTextHelpFormatter
from hmm_utility import load_tsv
from numpy.random import choice



def parse_args():
    #####################
    # START CODING HERE #
    #####################
    # Implement a simple argument parser (WITH help documentation!) that parses
    # the information needed by main() from commandline. Take a look at the
    # argparse documentation, the parser in hmm_utility.py or align.py
    # (from the Dynamic Programming exercise) for hints on how to do this.

    parser = ArgumentParser()
    # parser.add_argument(?)
    # parser.add_argument(?)
    # parser.add_argument(?)
    # parser.add_argument(?)
    
    #####################
    #  END CODING HERE  #
    #####################


def generate_sequence(A,E):
    #####################
    # START CODING HERE #
    #####################
    # Implement a function that generates a random sequence using the choice()
    # function, given a Transition and Emission matrix.
    
    # Look up its documentation online:
    # https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.random.choice.html
            
    sequence = '?'
    
    #####################
    #  END CODING HERE  #
    #####################
    
    return sequence



def main():
    args = parse_args()
    #####################
    # START CODING HERE #
    #####################
    # Uncomment and complete (i.e. replace '?' in) the lines below:
    
    # N = args.?               # The number of sequences to generate
    # out_file = args.?        # The file path to which to save the sequences
    # A = load_tsv(args.? )    # Transition matrix
    # E = load_tsv(args.? )    # Emission matrix
    # with open(out_file,'w') as f:
        # for i in range(N):
        #     seq = ?
        #     f.write('>random_sequence_%i\n%s\n' % (i,seq))
        
    #####################
    #  END CODING HERE  #
    #####################
    


if __name__ == "__main__":
    main()

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
    <Ville Lehtonen, Stud. nr.: 2658063, VUnetID: VLN490>
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

    parser = ArgumentParser(prog = 'python3 sequence_generator.py', formatter_class = RawTextHelpFormatter, 
    description =
    '  generate sequences from given transition and emission matrices \n\n'
    '  Example syntax:\n'
    '    python3 sequence_generator.py A.tsv E.tsv')
    # Positionals
    parser.add_argument('transition', help='path to a TSV formatted transition matrix')
    parser.add_argument('emission', help='path to a TSV formatted emission matrix')

    #Optionals
    parser.add_argument('-N', dest='sequences', type=int, default=200,
    help='the amount of sequences to be generated by the program, default = 200')

    parser.add_argument('-o', dest='out_file', default="output.txt",
        help='path to a file where output is saved\n')

    return parser.parse_args()


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
    
    transition_states = list(A.keys())
    emission_states = list(E.keys())
    alphabet = list(list(E.values())[0].keys()) ## symbols in E
    current_state = str(transition_states[0]) ## set to begin state
    sequence = ''

    while True:
        # transition
        transition_probs = list(A[current_state].values())
        if (sum(transition_probs) > 0.99):
            current_state = choice(transition_states, 1, p=transition_probs)[0]

            # emission
            if current_state in emission_states:
                emission_probs = list(E[current_state].values())
                emission = choice(alphabet, 1, p=emission_probs)[0]
                sequence += emission
        else:
            break

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
    
    N = args.sequences
    A = load_tsv(args.transition)
    E = load_tsv(args.emission)
    out_file = args.out_file
    
    with open(out_file, 'w') as f:
        for i in range(N):
            seq = generate_sequence(A, E)
            f.write('>random_sequence_%i\n%s\n' % (i,seq))
            print(seq)

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

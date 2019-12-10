#!/usr/bin/python3

"""
DESCRIPTION:
    Template code for the Hidden Markov Models assignment in the Algorithms in Sequence Analysis course at the VU.

INSTRUCTIONS:
    Complete the code (compatible with Python 3!) upload to CodeGrade via corresponding Canvas assignment.

AUTHOR:
    <Ville Lehtonen, Stud. nr.: 2658063, VUnetID: VLN490>
"""

import os.path as op

from os import makedirs
from math import log10
from hmm_utility import parse_args, load_fasta, load_tsv, print_trellis, print_params, serialize



def viterbi(X,A,E):
    """Given a single sequence, with Transition and Emission probabilities,
    return the most probable state path, the corresponding P(X), and trellis."""

    allStates = A.keys()
    emittingStates = E.keys()
    L = len(X) + 2

    # Initialize
    V = {k:[0] * L for k in allStates} # The Viterbi trellis
    V['B'][0] = 1.

    # Middle columns
    for i,s in enumerate(X):
        for l in emittingStates:
            terms = [V[k][i] * A[k][l] for k in allStates]
            V[l][i+1] = max(terms) * E[l][s]

    # Last column
    for k in allStates:
        term = V[k][i+1] * A[k]['E'] 
        if term > V['E'][-1]:
            V['E'][-1] = term
            pi = k # Last state of the State Path

    # FOR VITERBI ONLY: Trace back the State Path
    l = pi
    i = L-2
    while i:
        i -= 1
        for k in emittingStates:
            if V[k][i] * A[k][l] * E[l][X[i]] == V[l][i+1]:
                pi = k + pi
                l = k
                break

    P = V['E'][-1] # The Viterbi probability: P(X,pi|A,E)
    return(pi,P,V) # Return the state path, Viterbi probability, and Viterbi trellis

def forward(X,A,E):
    """Given a single sequence, with Transition and Emission probabilities,
    return the Forward probability and corresponding trellis."""

    allStates = A.keys()
    L = len(X) + 2

    # Initialize
    F = {k:[0] * L for k in allStates}
    F['B'][0] = 1

    #####################
    # START CODING HERE #
    #####################
    # HINT: The Viterbi and Forward algorithm are very similar! 
    # Adapt the viterbi() function to account for the differences.

    # Middle columns
    # for ...

    # Last columns
    # for ...:
    #     F['E'][-1] += ...
    emittingStates = E.keys()

    # Middle columns
    for i,s in enumerate(X):  ## creates an enumerate object with i = 1,2,3... and x = C,C,H,H,P,C,C...
        for l in emittingStates: ## dict_keys([L', 'D'])
            terms = [F[k][i] * A[k][l] for k in allStates] # k = (B,L,D,E), l = (L,D), i = (1,2,3...n) 
            F[l][i+1] = sum(terms) * E[l][s] ## s = (C,C,H,H,P,C,C)

    # Last column
    term = [F[k][i+1] * A[k]['E'] for k in allStates]
    F['E'][-1] = sum(term)

    #####################
    #  END CODING HERE  #
    #####################

    P = F['E'][-1] # The Forward probability: P(X|A,E)
    return(P,F)

def backward(X,A,E):
    """Given a single sequence, with Transition and Emission probabilities,
    return the Backward probability and corresponding trellis."""

    allStates = A.keys()
    emittingStates = E.keys()
    L = len(X) + 2 ## L = 12+2 = 14

    # Initialize
    B = {k:[0] * L for k in allStates} # The Backward trellis
    for k in allStates:
        B[k][-2] = A[k]['E']

    #####################
    # START CODING HERE #
    #####################
    # Remaining columns
    # for i in range(L-3,-1,-1):
    #     s = seq[i]
    #     ...
    
    for i in range(L-3, -1, -1): ## L = 14, i = 11,10,9,8,7....1,0
        s = X[i] ## s = H, C, H, H ... C, C
        for k in allStates: ## k = (B, L, D, E)
            terms = [A[k][l]*E[l][s]*B[l][i+1] for l in emittingStates] ## l = (L, D)
            B[k][i] = sum(terms)

    #####################
    #  END CODING HERE  #
    #####################

    P = B['B'][0] # The Backward probability -- should be identical to Forward!
    return(P,B)

def baumwelch(set_X,A,E):
    """Given a set of sequences X and priors A and E,
    return the Sum Log Likelihood of X given the priors,
    along with the calculated posteriors for A and E."""

    allStates = A.keys()
    emittingStates = E.keys()
    
    # Initialize a new (posterior) Transition and Emission matrix
    new_A = {}
    for k in A:
        new_A[k] = {l:0 for l in A[k]}

    new_E = {}
    for k in E:
        new_E[k] = {s:0 for s in E[k]}

    # Iterate through all sequences in X
    SLL = 0 # Sum Log-Likelihood
    for X in set_X:
        P,F = forward(X,A,E)  # Save both the forward probability and the forward trellis
        _,B = backward(X,A,E) # Forward P == Backward P, so only save the backward trellis
        SLL += log10(P)

        #####################
        # START CODING HERE #
        #####################
        # Inside the for loop: Expectation
        # Count how often you observe each transition and emission.
        # Add the counts to your posterior matrices. (new_A, new_E)
        # Remember to normalize to the sequence's probability P!

        #i  0 1 2 3 4 5 6 7 8 9 10 11 12 13
        #F: - C C H H P C C P H  H  C  H  -

        #i   0 1 2 3 4 5 6 7 8  9  10 11
        #X   C C H H P C C P H  H  C  H 

        for k in allStates:
            for l in emittingStates:
                for i,s in enumerate(X):
                    #emissions
                    new_E[l][s] += (F[l][i+1] * B[l][i+1] / P)
                    #transitions
                    new_A[k][l] += F[k][i] * A[k][l] * E[l][s] * B[l][i+1] / P
                
        #transitions for the last state E
        for k in allStates:
            new_A[k]['E'] = (F[k][-2] * A[k]['E'] / P)

    # Outside the for loop: Maximization
    # Normalize row sums to 1 (except for one row in the Transition matrix!)
    # new_A = ...
    # new_E = ...

    ## CHECK OUT HOW TO NORMALIZE !!!! (Returning key error now)

    for l in emittingStates:
        sumOfValues = sum(new_E[l].values())
        for emission, prob in new_E[l].items():
            new_E[l][emission] = (prob / sumOfValues)

    for k in allStates:
        sumOfValues = sum(new_A[k].values())
        if sumOfValues > 0:
            for transition, prob in new_A[k].items():
                new_A[k][transition] = (prob / sumOfValues)

    #####################
    #  END CODING HERE  #
    #####################

    return(SLL,new_A,new_E)

        ## A: {'B': {'B': 0.0, 'L': 0.5, 'D': 0.5, 'E': 0.0},
        #      'L': {'B': 0.0, 'L': 0.7, 'D': 0.2, 'E': 0.1},
        #      'D': {'B': 0.0, 'L': 0.2, 'D': 0.7, 'E': 0.1},
        #      'E': {'B': 0.0, 'L': 0.0, 'D': 0.0, 'E': 0.0}}
        
        # E = {'L': {'H': 0.5, 'P': 0.0, 'C': 0.5},'D': {'H': 0.0, 'P': 0.5, 'C': 0.5}}

        #P:  4.385861279296873e-08

        #F: {'B': [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #    'L': [0, 0.25, 0.11249999999999999, 0.05062499999999999, 0.017718749999999995, 0.0, 0.00017718749999999996, 0.00012403124999999996, 0.0, 9.457382812499998e-06, 3.310083984374999e-06, 1.1585293945312497e-06, 4.385861279296873e-07, 0],
        #    'D': [0, 0.25, 0.11249999999999999, 0.0, 0.0, 0.0017718749999999996, 0.0006201562499999998, 0.00023477343749999993, 9.457382812499997e-05, 0.0, 0.0, 3.3100839843749994e-07, 0.0, 0],
        #    'E': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4.385861279296873e-08]}

        #B: {'B': [4.385861279296874e-08, 9.746358398437497e-08, 2.1658574218749995e-07, 6.188164062499999e-07, 6.188164062499998e-06, 2.3477343749999994e-05, 5.217187499999999e-05, 0.00011593749999999999, 0.0011593749999999998, 0.0033125, 0.01125, 0.025, 0.0, 0],
        #    'L': [4.624105595703124e-08, 1.1479044335937496e-07, 3.0322003906249993e-07, 8.663429687499998e-07, 2.4752656249999995e-06, 1.7506562499999997e-05, 3.2462499999999996e-05, 4.6375e-05, 0.0016231249999999996, 0.004637499999999999, 0.01325, 0.034999999999999996, 0.1, 0], 
        #    'D': [3.270444707031249e-08, 6.064400781249999e-08, 8.663429687499999e-08, 2.4752656249999996e-07, 8.663429687499997e-06, 2.4752656249999993e-05, 6.144687499999998e-05, 0.00016231249999999997, 0.00046374999999999997, 0.001325, 0.007, 0.010000000000000002, 0.1, 0], 
        #    'E': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0]}

        # SLL: -7.357945108770817

        # new_E: {'L': {'H': 0, 'P': 0, 'C': 0},
        #         'D': {'H': 0, 'P': 0, 'C': 0}}

        # new_A: {'B': {'B': 0, 'L': 0, 'D': 0, 'E': 0},
        #         'L': {'B': 0, 'L': 0, 'D': 0, 'E': 0}, 
        #         'D': {'B': 0, 'L': 0, 'D': 0, 'E': 0}, 
        #         'E': {'B': 0, 'L': 0, 'D': 0, 'E': 0}}

        #set_X: ['CCHHPCCPHHCH']Add the contribution of sequence j to A and E


def main(args = False):
    "Perform the specified algorithm, for a given set of sequences and parameters."
    
    # Process arguments and load specified files
    if not args: args = parse_args()

    cmd = args.command            # viterbi, forward, backward or baumwelch
    verbosity = args.verbosity
    set_X, labels = load_fasta(args.fasta)  # List of sequences, list of labels
    A = load_tsv(args.transition) # Nested Q -> Q dictionary
    E = load_tsv(args.emission)   # Nested Q -> S dictionary
    
    def save(filename, contents):
        if args.out_dir:
            makedirs(args.out_dir, exist_ok=True) # Make sure the output directory exists.
            path = op.join(args.out_dir,filename)
            with open(path,'w') as f: f.write(contents)
        # Note this function does nothing if no out_dir is specified!



    # VITERBI
    if cmd == 'viterbi':
        for j,X in enumerate(set_X): # For every sequence:
            # Calculate the most probable state path, with the corresponding probability and matrix
            Q, P, T = viterbi(X,A,E)

            # Save and/or print relevant output
            label = labels[j]
            save('%s.path' % label, Q)
            save('%s.matrix' % label, serialize(T,X))
            save('%s.p' % label, '%1.2e' % P)
            print('>%s\n Path = %s' % (label,Q))
            if verbosity: print(' Seq  = %s\n P    = %1.2e\n' % (X,P))
            if verbosity >= 2: print_trellis(T, X)
            


    # FORWARD or BACKWARD
    elif cmd in ['forward','backward']:
        if cmd == 'forward':
            algorithm = forward
        elif cmd == 'backward':
            algorithm = backward

        for j,X in enumerate(set_X): # For every sequence:
            # Calculate the Forward/Backward probability and corresponding matrix
            P, T = algorithm(X,A,E)

            # Save and/or print relevant output
            label = labels[j]
            save('%s.matrix' % label, serialize(T,X))
            save('%s.p' % label, '%1.2e' % P)
            if verbosity >= 2:
                print('\n>%s\n P = %1.2e\n' % (label,P))
                print_trellis(T, X)
            elif verbosity: print('>%-10s\tP = %1.2e' % (label,P))



    # BAUM-WELCH TRAINING
    elif cmd == 'baumwelch':
        # Initialize
        i = 1
        i_max = args.max_iter
        threshold = args.conv_thresh

        current_SLL, A, E = baumwelch(set_X,A,E)
        if verbosity: print('Iteration %i, prior SLL = %1.2e' % (i,current_SLL))
        if verbosity >= 2: print_params(A,E)
        
        last_SLL = current_SLL - threshold - 1 # Iterate at least once

        # Iterate until convergence or limit
        while i < i_max and current_SLL - last_SLL > threshold:
            i += 1
            last_SLL = current_SLL

            # Calculate the Sum Log-Likelihood of X given A and E,
            # and update the estimates (posteriors) for A and E.
            current_SLL, A, E = baumwelch(set_X,A,E)

            if verbosity: print('Iteration %i, prior SLL = %1.2e' % (i,current_SLL))
            if verbosity >= 2: print_params(A,E)

        converged = current_SLL - last_SLL <= threshold
        final_SLL = sum([log10(forward(X,A,E)[0]) if forward(X,A,E)[0] > 0 else 0 for X in set_X])

        # Save and/or print relevant output
        save('SLL','%1.2e\t%i\t%s' % (final_SLL, i, converged))
        save('posterior_A',serialize(A))
        save('posterior_E',serialize(E))
        if verbosity: print('========================================\n')

        if converged:
            print('Converged after %i iterations.' % i)
        else:
            print('Failed to converge after %i iterations.' % i_max)

        if verbosity:
            print('Final SLL: %1.2e' % final_SLL)
            print('Final parameters:')
            print_params(A,E)



if __name__ == '__main__':
	main()
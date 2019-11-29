from argparse import ArgumentParser, RawTextHelpFormatter


#########################
### UTILITY FUNCTIONS ###
#########################
# A lot of these are used by hmm.main();
# skim these to understand what's going on,
# but don't code in this file!

def parse_args():
    "Parses inputs from commandline and returns them as a Namespace object."

    parser = ArgumentParser(prog = 'python3 hmm.py',
        formatter_class = RawTextHelpFormatter, description =
        '  Perform the specified algorithm, with given sequences and parameters.\n\n'
        '  Example syntax:\n'
        '    python3 hmm.py -vv viterbi seq.fasta A.tsv E.tsv\n'
        '    python3 hmm.py baumwelch in.fa priorA priorE -o ./outputs -i 1')

    # Positionals
    parser.add_argument('command', help='which algorithm to run',
        choices=['viterbi','forward','backward','baumwelch'])
    parser.add_argument('fasta', help='path to a FASTA formatted input file')
    parser.add_argument('transition', help='path to a TSV formatted transition matrix')
    parser.add_argument('emission', help='path to a TSV formatted emission matrix')

    # Optionals
    parser.add_argument('-v', '--verbose', dest='verbosity', action='count', default=0,
        help='print verbose output specific to the algorithm\n'
             '  (print even more output if flag is given twice)')

    parser.add_argument('-o', dest='out_dir',
        help='path to a directory where output files are saved\n'
             '  (directory will be made if it does not exist)\n'
             '  (file names and contents depend on algorithm)')
    parser.add_argument('-i', dest='max_iter', type=int, default=100,
        help='maximum number of iterations (Baum-Welch only, default: 100 )')
    parser.add_argument('-c', dest='conv_thresh', type=float, default=0.01,
        help='convergence threshold        (Baum-Welch only, default: 0.01)\n ')

    return parser.parse_args()

def load_fasta(path):
    """Load a FASTA formatted set of sequences. Returns two lists: sequences and labels.
    Warning: Will likely throw errors if the file is not FASTA formatted!"""
    labs = []
    seqs = []
    with open(path) as f:
        for line in f:
            if line.startswith('>'):
                labs.append(line.strip()[1:])
                seqs.append('')
            else:
                seqs[-1] += line.strip()
    return seqs, labs

def load_tsv(path):
    "Load a TSV formatted set of (prior) parameters. Return as a nested dictionary."
    out = {}
    with open(path) as f:
        header = f.readline().strip().split('\t') # Read, strip and split the header line
        for line in f:
            ls = line.rstrip().split('\t')
            out[ls[0]] = {header[i]:float(v) for i,v in enumerate(ls[1:])}
    return out

def print_trellis(T,sequence):
    "Pretty print function for a Viterbi/Forward/Backward dynamic programming matrix."
    Q = sort_states(T.keys())
    X = '-' + sequence + '-'
    print('   '+''.join(['%-8s ' % s for s in X]))
    for q in Q:
        print('%2s ' % q + ''.join(['%1.2e ' % p for p in T[q]]))
    print('')

def print_params(A,E):
    "Pretty print function for the Transition matrix (from a nested dictionary)."
    QA = sort_states(A.keys())
    print('\n[A]   ' + ''.join('%-5s ' % j for j in QA))
    for i in QA:
        print('%5s ' % i + ''.join('%0.3f ' % A[i][j] for j in QA))
    QE = sorted(E.keys())
    S = sorted(E[QE[0]].keys())
    print('\n[E]   ' + ''.join('%-5s ' % s for s in S))
    for i in QE:
        print('%5s ' % i + ''.join('%0.3f ' % E[i][s] for s in S))
    print('')

def serialize(dictionary, sequence=False):
    keys = sorted(dictionary.keys())
    if sequence:      # Trellis
        keys = sort_states(keys)
        ix   = range(len(sequence)+2)
        out  = ['\t'.join(list(' -' + sequence + '-'))]
    elif 'B' in keys: # Transition matrix
        keys = sort_states(keys)
        ix   = keys
        out  = ['\t'.join([' ']+keys)]
    else:             # Emission matrix
        ix  = sorted(dictionary[keys[0]].keys())
        out = ['\t'.join([' ']+ix)]
    for k in keys:
        line = k + '\t' + '\t'.join(['%1.2e' % dictionary[k][i] for i in ix])
        out.append(line)
    return '\n'.join(out)

def sort_states(states):
    "Sort a list of states, while making sure 'B' and 'E' respectively start and end the list."
    Q = sorted(states)
    Q.remove('B')
    Q.remove('E')
    return ['B'] + Q + ['E']

################################
### END OF UTILITY FUNCTIONS ###
################################
import sys


def fasta2seqs(file):
    """
    Takes a fasta file and converts the contents to single sequence strings.

    :param file: the fasta file
    :return: a list of sequence strings
    """
    seqs = []
    seq = ""
    with open(file) as f:
        for line in f:
            if line[0] == ">" or line[0] == ";":
                if len(seq) > 0:
                    seqs.append(seq)
                seq = ""
            else:
                seq += line.strip()
        seqs.append(seq)
    return seqs


def main():
    src_file = sys.argv[1]

    seqs = fasta2seqs(src_file)
    for seq in seqs:
        print(seq)


if __name__ == "__main__":
    main()

import os
import sys
import argparse
from clip_similarity import ClipSimilarity



def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--root_dir', type=str)
    args = parser.parse_args()

    sim_evaluator = ClipSimilarity()
    import pdb; pdb.set_trace()


if __name__ == '__main__':
    main()

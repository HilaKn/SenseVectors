import argparse
from os import listdir
from os.path import isfile, join
import os
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Generate adjectives with multi senses list.')

    parser.add_argument('input_folder',help='input folder path - adjectives clusters')
    parser.add_argument('output_file',help='output file for listing all the multi sense adjectives')
    args = parser.parse_args()


    onlyfiles = [os.path.splitext(f)[0] for f in listdir(args.input_folder) if isfile(join(args.input_folder, f))]

    with open(args.output_file, 'w') as f:
        f.writelines("\n".join(onlyfiles))

import cv2
import os
#import argparse
import sys
import getopt

# remove 1st argument from list of arguments
argumentList = sys.argv[1:]

# options
options = "hior:"

# long options
long_options = ["help", "indir", "outdir", "resolution"]

# declare variables
indirectory =
outdirectory =
resolution =

try:
    # parsing arguments
    arguments, values = getopt.getopt(argumentList, options, long_options)

    # checking for each argument
    for currentArgument, currentValue in arguments:
        if currentArgument in("-h", "--help"):
            print(
            """
            usage: resize.py [-h] [--128 [128]] [--256 [256]] [--512 [512]] [--1024 [1024]] i [i ...] o [o ...]

            Specify files

            positional arguments:
              i              Specify a directory with files
              o              Specify the destination for modified files

            optional arguments:
              -h, --help     show this help message and exit
              --128 [128]    Modify images to 128x128 resolution
              --256 [256]    Modify images to 256x256 resolution
              --512 [512]    Modify images to 512x512 resolution
              --1024 [1024]  Modify images to 1024x1024 resolution
            """
            )
        if currentArgument in ("-i", "--indir"):
            print('Gathering from: ', sys.argv[0])
            indirectory = sys.argv[0]
        if currentArgument in ("-o", "--outdir"):
            print('Outputting to: ', sys.argv[1])
            outdirectory = sys.argv[1]
        if currentArgument in ("-r", "--resolution"):
            print('Scaling to: ', sys.argv[2])
            resolution = sys.argv[2]
except:
    print(str(err))

"""
parser = argparse.ArgumentParser(description='Specify files')
parser.add_argument('infolder', metavar = 'infolder', type = str, nargs = '+', help = 'Specify a directory with files')
parser.add_argument('outfolder', metavar = 'outfolder', type = str, nargs = '+', help = 'Specify the destination for modified files')
parser.add_argument('--128', const = '128x128', default = max, nargs = '?', help = 'Modify images to 128x128 resolution')
parser.add_argument('--256', const = '256x256', default = max, nargs = '?', help = 'Modify images to 256x256 resolution')
parser.add_argument('--512', const = '512x512', default = max, nargs = '?', help = 'Modify images to 512x512 resolution')
parser.add_argument('--1024', const = '1024x1024', default = max, nargs = '?', help = 'Modify images to 1024x1024 resolution')

args = parser.parse_args()
print('Gathering from: ' + args.accumulate(args.infolder))
print('Outputting to: ' + args.accumulate(args.outfolder))

indirectory = args.accumulate(args.infolder)
outdirectory = args.accumulate(args.outfolder)
"""


for filename in os.listdir(indirectory):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        print(os.path.join(directory, filename))

        # read in image
        img = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
        print('Original dimensions: ', img.shape)

        width =
        height =
        dim = (width, height)
        # resize the image now
        resized = cv2.resize(img, dim, interpolation = cv2.INNER_AREA)

    else:
        continue

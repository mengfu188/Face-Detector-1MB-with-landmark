from pathlib import Path
import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-root', default='/home/cmf/datasets/wider_face/WIDER_train/images')
    # parser.add_argument('--input-size')
    parser.add_argument('--count', type=int, default=6000)
    parser.add_argument('--output-file', default='dataset.txt')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    root = Path(args.image_root)
    files = list(root.glob('*/*'))
    f = open(args.output_file, 'w')
    for file in files[:args.count]:
        f.write(str(file.absolute()) + '\n')
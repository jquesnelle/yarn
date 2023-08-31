import argparse
import numpy
import json

def main(args):
    obj = json.load(open(args.file, "r", encoding="utf-8"))
    results = [result["acc"] for result in obj["results"].values()]
    print(numpy.average(results))

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("file", type=str)
    main(args.parse_args())
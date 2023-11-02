import argparse
import matplotlib.pyplot as plt
import pandas as pd

def main(args):
    data = pd.read_csv(args.csv)
    fig, ax = plt.subplots(figsize=(10,5))

    x_data = [float(x) for x in data.columns[1:]]
    for row in data.values:
        label = row[0].replace("NousResearch/", "")
        ax.plot(x_data, [float(x) for x in row[1:]], label=label)

    ax.set_xlabel("Context Window")
    ax.set_ylabel("Perplexity (lower is better)")

    ax.set_xlim(args.xmin, args.xmax)
    ax.set_ylim(args.ymin, args.ymax)

    ax.legend(loc="upper right")

    fig.savefig(args.csv + ".png")
    fig.savefig(args.csv + ".pdf", transparent=True)

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("csv", type=str)
    args.add_argument("--xmin", type=int, default=0)
    args.add_argument("--xmax", type=int, default=131072)
    args.add_argument("--ymin", type=float, default=2.2)
    args.add_argument("--ymax", type=float, default=3.8)
    main(args.parse_args())
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

    ax.set_xlim(0, 131072)
    ax.set_ylim(2.2, 3.8)

    ax.legend(loc="upper right")

    fig.savefig(args.csv + ".png")
    fig.savefig(args.csv + ".pdf", transparent=True)

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("csv", type=str)
    main(args.parse_args())
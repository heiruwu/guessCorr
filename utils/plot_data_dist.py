import argparse
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({'figure.figsize':(7,5), 'figure.dpi':100})

def plot_data_dist(csv_file):
    correlations_frame = pd.read_csv(csv_file)
    plt.hist(correlations_frame['corr'], bins=100)
    plt.gca().set(title='data distribution', xlabel='corr', ylabel='count')
    plt.savefig('materials/dist.png')

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('-c', '--csv_file', required=True, type=str,
                      help='csv file path (required: True)')

    args = args.parse_args()
    plot_data_dist(args.csv_file)

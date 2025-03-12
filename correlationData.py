from printData import correlation_graph

from printData import print_data


def correlation_data(df):
    print_data(df)
    spearman_corr = df.corr(method='spearman')
    print(spearman_corr)
    correlation_graph(spearman_corr)

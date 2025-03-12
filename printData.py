from matplotlib import pyplot as plt
import seaborn as sns

def correlation_graph(spearman_corr):
    plt.figure(figsize=(28, 25))
    sns.heatmap(spearman_corr, annot=True, fmt=".2f", cmap='coolwarm', cbar=True)
    plt.title('Spearman Correlation Matrix')
    plt.show()

    correlation = spearman_corr['Accident'].drop(index= 'Accident', axis =1 )
    print(correlation)

    plt.figure(figsize= (20, 20))
    plt.xticks(rotation='vertical',  fontsize =20)
    plt.yticks(fontsize=20)
    sns.barplot(x=correlation.index, y=correlation.values)
    plt.show()

def alpha_graph(ccp_alphas, node_counts, depth):
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(ccp_alphas, node_counts, marker="o", drawstyle="steps-post")
    ax[0].set_xlabel("alpha")
    ax[0].set_ylabel("number of nodes")
    ax[0].set_title("Number of nodes vs alpha")
    ax[1].plot(ccp_alphas, depth, marker="o", drawstyle="steps-post")
    ax[1].set_xlabel("alpha")
    ax[1].set_ylabel("depth of tree")
    ax[1].set_title("Depth vs alpha")
    fig.tight_layout()
    plt.show()

def print_data(df):
    print(df.info())
    print("\n")
    print(df.head(10))

def plot_matrix(matrix):
    labels = [False, True]
    sns.heatmap(matrix, annot=True, fmt=".2f", cmap='Greens', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Prediction')
    plt.ylabel('Ground Truth')
    plt.show()
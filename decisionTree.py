from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import plot_tree, DecisionTreeClassifier
from sklearn import metrics, tree
from sklearn.metrics import classification_report, confusion_matrix
from printData import *
from imblearn.over_sampling import SMOTE


class RandomForestTree:
    def __init__(self, df):
        self.df = df
        self.train_X = None
        self.test_X = None
        self.train_Y = None
        self.test_Y = None
        self.pred_Y = None
        self.model = None
        self.alpha = None

    def training_data(self):
        X = self.df.drop(['Accident'], axis=1)
        Y = self.df['Accident']
        self.train_X, self.test_X, self.train_Y, self.test_Y = train_test_split(X, Y, test_size=0.3, random_state=1)
        self.post_pruning()

        smote = SMOTE(random_state=3)
        self.train_X, self.train_Y = smote.fit_resample(self.train_X, self.train_Y)

        rus = RandomUnderSampler(random_state=42)
        self.train_X, self.train_Y = rus.fit_resample(self.train_X, self.train_Y)
        self.model = RandomForestClassifier(
            random_state=3,
            class_weight='balanced',
            ccp_alpha=self.alpha,
            max_depth=6,
            max_leaf_nodes= 25,
            min_samples_leaf=6,
            min_samples_split=4,
            n_estimators=120,
            criterion='entropy',
        )

        self.model.fit(self.train_X, self.train_Y)

    def predict(self):
        if self.model is None:
            raise Exception('model not trained')

        else:
            self.pred_Y = self.model.predict(self.test_X)

    def print_tree(self):
        tree_to_plot = self.model.estimators_[0]
        plt.figure(figsize=(80, 45))
        plot_tree(tree_to_plot, feature_names=self.df.columns.tolist(), filled=True, rounded=True, fontsize=10)
        plt.title("Decision Tree from Random Forest")
        plt.show()

    def accuracy(self):
        print('Training Accuracy : ',
              metrics.accuracy_score(self.train_Y,
                                     self.model.predict(self.train_X)) * 100)
        print('Validation Accuracy : ',
              metrics.accuracy_score(self.test_Y,
                                     self.pred_Y) * 100)

    def post_pruning(self):
        clf = DecisionTreeClassifier(random_state=78)
        path = clf.cost_complexity_pruning_path(self.train_X, self.train_Y)  # Отримати можливі значення alpha
        ccp_alphas, impurities = path.ccp_alphas, path.impurities
        clfs = []
        for ccp_alpha in ccp_alphas:
            clf = DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
            clf.fit(self.train_X, self.train_Y)
            clfs.append(clf)
        print(
            "Number of nodes in the last tree is: {} with ccp_alpha: {}".format(
                clfs[-1].tree_.node_count, ccp_alphas[-1]
            )
        )
        self.alpha = round(ccp_alphas[-1] / 1.5, 5)
        clfs = clfs[:-1]
        ccp_alphas = ccp_alphas[:-1]

        node_counts = [clf.tree_.node_count for clf in clfs]
        depth = [clf.tree_.max_depth for clf in clfs]
        alpha_graph(ccp_alphas, node_counts, depth)

    def inference(self):
        report =classification_report(self.test_Y, self.pred_Y)
        matrix = confusion_matrix(self.test_Y, self.pred_Y)
        print('\n Test Report\n')
        print(report)
        # plots confusion matrix
        plot_matrix(matrix)



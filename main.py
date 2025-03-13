from sklearn.utils.multiclass import type_of_target
from preprocessingData import *
from correlationData import *
from decisionTree import *

file_path = "dataset_traffic_accident_prediction1.csv"

# Load the latest version
df = pd.read_csv(file_path)

#print data
print_data(df)

# One hot enocding
df = preprocessing(df)

#print correlation
correlation_data(df)

#Random Forest algorithm
rfrTree = RandomForestTree(df)
rfrTree.training_data()
rfrTree.predict()
rfrTree.print_tree()
rfrTree.accuracy()
rfrTree.inference()


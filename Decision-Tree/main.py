# -------------------------------------------------------------------------
# AUTHOR: Suhuan Pan
# FILENAME: title of the source file: decision_tree.py
# SPECIFICATION: use python to read a csv file and plot a decision tree
# FOR: CS 4210 - Assignment #1
# TIME SPENT: 5 hours
# -----------------------------------------------------------*/
# IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH
# AS numpy OR pandas. You have to work here only with standard
# dictionaries, lists, and arrays
# importing some Python libraries
from sklearn import tree
import matplotlib.pyplot as plt
import csv

db = []
X = []
Y = []

# reading the data in a csv file
with open('contact_lens.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for i, row in enumerate(reader):
        if i > 0:  # skipping the header
            db.append (row)
            print(row)

# transform the original categorical training features
# into numbers and add to the 4D array X:
# Feature 1 Age's values:
# Young = 0, Prepresbyopic = 1, Presbyopic = 2
# young = 0, prepresbyopic = 0.3, presbyopic = 1
# Feature 2 Spectacle Prescription's values:
# Myope = 0, Hypermetrope = 1
# Feature 3 Astigmatism's values
# No = 1, Yes = 0
# Feature 4 Tear Production Rate's values:
# Reduced = 0, Normal = 1

X = [[0, 0, 1, 0],
     [1, 0, 1, 1],
     [2, 0, 1, 0],
     [2, 0, 1, 1],
     [1, 0, 0, 1],
     [0, 0, 0, 1],
     [0, 1, 1, 0],
     [2, 0, 0, 0],
     [1, 1, 1, 0],
     [0, 0, 0, 0]]

# transform the original categorical training classes into numbers
# and add to the vector Y.
# Recommended Lenses: No = 1, Yes = 0
Y = [1, 1, 1, 0, 0, 0, 1, 1, 1, 0]


# len(y) = 4
# fitting the decision tree to the data
clf = tree.DecisionTreeClassifier(criterion = 'entropy')
clf = clf.fit(X, Y)

# plotting the decision tree
tree.plot_tree(clf, feature_names=['Age',
                                   'Spectacle',
                                   'Astigmatism',
                                   'Tear Production'],
               class_names=['Yes', 'No'], # node_ids=True,
               filled=True, rounded=True)

plt.show()
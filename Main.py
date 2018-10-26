import StringIO
import os

import numpy as np
import pydotplus

from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


def make_pdf(percent, m_classifier):
    dot_data = StringIO.StringIO()
    tree.export_graphviz(m_classifier, out_file=dot_data)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_png("graph" + str(percent) + ".png")


experiments = [
    [1, 60, 40],
    [2, 70, 30],
    [3, 80, 20]
]

for experiment in experiments:
    trainPercent = experiment[1]

    print "Experiment : #", experiment[0], "trainPercent =", trainPercent, "%"

imported_data = []

for line in open('tae.data', 'r'):
    line = line.split(",")

    newline = []
    for item in line:
        if item == "?":
            newline.append(0)
        else:
            newline.append(item)

    line = map(int, newline)
    imported_data.append(line)

all_data = np.array(imported_data)

X = all_data[:, :5]
y = all_data[:, 4]

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=trainPercent / 100.0)
clf = DecisionTreeClassifier().fit(X_train, y_train)
y_predicted = clf.predict(X_test)

print y_predicted
guessed_num = 0

for ind in range(0, len(y_predicted)):
    if y_predicted[ind] == y_test[ind]:
        guessed_num += 1

print "TOTAL:", len(y_test)
print "guessed:", guessed_num
print "accuracy:", 1.0 * guessed_num / len(y_test)
print ""
make_pdf(trainPercent, clf)

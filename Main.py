import StringIO

import numpy as np
import pydot
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import pydotplus


def make_pdf(percent, m_classifier):
    dot_data = StringIO.StringIO()
    tree.export_graphviz(m_classifier, out_file=dot_data)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_png("graph" + str(percent) + ".png")


experiments = [
    [1, 60, 40],
    [2, 70, 30],
    [3, 80, 20],
    [4, 90, 10]
]

for experiment in experiments:
    trainPercent = experiment[1]

    print "Experiment: #", experiment[0], ", trainPercent =", trainPercent, "%"
    data = []

    for line in open('tae.data', 'r'):
        line = line.split(",")
        line = map(float, line)
        data.append(line)

    allData = np.array(data)

    x = allData[:, :7]
    y = allData[:, 8]

    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=trainPercent / 100.0)

    # Train

    classifier = DecisionTreeClassifier(max_depth=5).fit(x_train, y_train)

    y_predicted = classifier.predict(x_test)

    guessed_num = 0

    for i in range(0, len(y_predicted)):
        if y_predicted[i] == y_test[i]:
            guessed_num += 1

    print "Total:", len(y_test)
    print "Guessed:", guessed_num
    print "Percent:", 100.0 * guessed_num / len(y_test)
    print ""
    make_pdf(trainPercent, classifier)

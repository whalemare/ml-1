import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

experiments = [
    [1, 60, 40],
    [2, 70, 30],
    [3, 80, 20],
    [4, 90, 10]
]

data = []

for line in open('tae.data', 'r'):
    line = line.split(",")
    line = map(int, line)
    data.append(line)

allData = np.array(data)

x = allData[:, :4]
y = allData[:, 5]

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.6)

print len(y)
print len(y_train)
print len(y_test)


# Train
# (Признаки, целевые классы)
# classifier = DecisionTreeClassifier().fit(x_train, y_train)

# укажем длину (глубину) дерева
classifier = DecisionTreeClassifier(max_depth=5).fit(x_train, y_train)

# Предсказать классы для x_test
y_predicted = classifier.predict(x_test)

print y_predicted

# Оценим точность предсказания

guessed_num = 0

for i in range(0, len(y_predicted)):
    if y_predicted[i] == y_test[i]:
        guessed_num += 1

print "Total:", len(y_test)
print "Guessed:", guessed_num
print "Percent:", 100.0 * guessed_num / len(y_test)

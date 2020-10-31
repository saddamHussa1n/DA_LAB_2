import scipy.stats as ss
import numpy as np
import csv
import matplotlib.pyplot as plt

with open("var11.csv", "r") as fin:
    reader = csv.reader(fin, delimiter=';')
    buf = []
    for row in reader:
        buf.append(row)
    buf.pop(0)
    for item in buf:
        for i in range(len(item)):
            n = item[i].replace(',', '.')
            item[i] = float(n)
    dataset = np.array(buf)
sample_mean = []
buf1 = []
buf2 = []
buf3 = []
buf4 = []
for i in range(400):
    buf1.append(dataset[i][0])
for i in range(400):
    buf2.append(dataset[i][1])
for i in range(400):
    buf3.append(dataset[i][2])
for i in range(400):
    buf4.append(dataset[i][3])
x1 = np.array(buf1)
x2 = np.array(buf2)
x3 = np.array(buf3)
x4 = np.array(buf4)
sample_mean.append(np.mean(x1))
sample_mean.append(np.mean(x2))
sample_mean.append(np.mean(x3))
sample_mean.append(np.mean(x4))
print("Вектор мат ожидания:", sample_mean)
print()
covariation = np.cov(dataset, rowvar=False, ddof=1)
print("Ковариационная матрица:")
print(covariation)
print()
std_vector = []
std_vector.append(np.std(x1, ddof=1))
std_vector.append(np.std(x2, ddof=1))
std_vector.append(np.std(x3, ddof=1))
std_vector.append(np.std(x4, ddof=1))
print("Коэффициенты корреляции:")
print("r(X1, X2) =", covariation[0][1]/(std_vector[0]*std_vector[1]))
print("r(X1, X3) =", covariation[0][2]/(std_vector[0]*std_vector[2]))
print("r(X1, X4) =", covariation[0][3]/(std_vector[0]*std_vector[3]))

plt.hist(x=x1, rwidth=0.87)
plt.title("Histogram for X1")
plt.show()

plt.hist(x=x2, rwidth=0.87)
plt.title("Histogram for X2")
plt.show()

plt.hist(x=x3, rwidth=0.87)
plt.title("Histogram for X3")
plt.show()

plt.hist(x=x4, rwidth=0.87)
plt.title("Histogram for X4")
plt.show()

print()
print("Тесты Колмагорова-Сморнова: ")
print("X1:\t", ss.kstest(x1, 'norm', (sample_mean[0], std_vector[0])))
print("X2:\t", ss.kstest(x2, 'norm', (sample_mean[1], std_vector[1])))
print("X3:\t", ss.kstest(x3, 'norm', (sample_mean[2], std_vector[2])))
print("X4:\t", ss.kstest(x4, 'norm', (sample_mean[3], std_vector[3])))

print()
print("Тесты Шапира-Уилка: ")
print("X1:\t", ss.shapiro(x1))
print("X2:\t", ss.shapiro(x2))
print("X3:\t", ss.shapiro(x3))
print("X4:\t", ss.shapiro(x4))



x = np.linspace(np.amin(x1), np.amax(x1), 100)
y = np.linspace(np.amin(x2), np.amax(x2), 100)
X1, X2 = np.meshgrid(x, y)
X1.transpose()
X2.transpose()
pos = np.dstack((X1, X2))
cov12 = [[covariation[0][0], covariation[0][1]], [covariation[1][0], covariation[1][1]]]
rv = ss.multivariate_normal([sample_mean[0], sample_mean[1]], cov12)
plt.contourf(X1, X2, rv.pdf(pos), levels=10)

plt.scatter(x1, x2)
plt.title("Scatter diagram for X1 and X2")
plt.xlabel("X1")
plt.ylabel("X2")
plt.show()


x = np.linspace(np.amin(x1), np.amax(x1), 100)
y = np.linspace(np.amin(x3), np.amax(x3), 100)
X1, X3 = np.meshgrid(x, y)
X1.transpose()
X3.transpose()
pos = np.dstack((X1, X3))
cov13 = [[covariation[0][0], covariation[0][2]], [covariation[2][0], covariation[2][2]]]
rv = ss.multivariate_normal([sample_mean[0], sample_mean[2]], cov13)
plt.contourf(X1, X3, rv.pdf(pos), levels=10)

plt.scatter(x1, x3)
plt.title("Scatter diagram for X1 and X3")
plt.xlabel("X1")
plt.ylabel("X3")
plt.show()

x = np.linspace(np.amin(x1), np.amax(x1), 100)
y = np.linspace(np.amin(x4), np.amax(x4), 100)
X1, X4 = np.meshgrid(x, y)
X1.transpose()
X4.transpose()
pos = np.dstack((X1, X4))
cov14 = [[covariation[0][0], covariation[0][3]], [covariation[3][0], covariation[3][3]]]
rv = ss.multivariate_normal([sample_mean[0], sample_mean[3]], cov14, allow_singular=True)
plt.contourf(X1, X4, rv.pdf(pos), levels=10)

plt.scatter(x1, x4)
plt.title("Scatter diagram for X1 and X4")
plt.xlabel("X1")
plt.ylabel("X4")
plt.show()



buf = []
for i in range(400):
    y1 = 5*x1[i] + 9*x3[i]
    y2 = 6*x1[i] + 10*x4[i]
    buf.append([y1, y2])
new_set = np.array(buf)

buf1 = []
buf2 = []
for i in range(400):
    buf1.append(new_set[i][0])
for i in range(400):
    buf2.append(new_set[i][1])
y1 = np.array(buf1)
y2 = np.array(buf2)
new_sample_mean = [np.mean(y1), np.mean(y2)]
new_sample_mean = np.array(new_sample_mean)
new_covariance = np.cov(new_set, rowvar=False, ddof=1)

a = np.array([[2, 0, 2, 0], [3, 0, 0, 1]])
at = np.array([[2, 0], [3, 0], [2, 0], [0, 1]])

expected_value = a.dot(np.array(sample_mean))
buf = a.dot(covariation)
theoretic_covariance = buf.dot(at)

print()
print("Эмпирически вычесленное мат ожидание:", new_sample_mean)
print("Мат ожидание по формулам:", expected_value)

print()
print("Эмпирически вычесленная ковариация: ")
print(new_covariance)

print()
print("Ковариация по формулам: ")
print(theoretic_covariance)





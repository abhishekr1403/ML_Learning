from math import isclose

import numpy as np
import pandas as pd
import math
from sklearn import linear_model


def using_sklearn():
    df = pd.read_csv('test_scores.csv')
    m = linear_model.LinearRegression()
    m.fit(df[['math']], df['cs'])
    return m.coef_, m.intercept_


def gradient_descent(x, y):
    m_curr = b_curr = 0
    iterations = 1000000
    n = len(x)
    learning_rate = 0.0002
    cost_previous = 0

    for i in range(iterations):
        y_predicted = m_curr * x + b_curr
        cost = (1 / n) * sum([val ** 2 for val in (y - y_predicted)])

        dm = -(2 / n) * sum(x * (y - y_predicted))
        db = -(2 / n) * sum((y - y_predicted))

        m_curr = m_curr - learning_rate * dm
        b_curr = b_curr - learning_rate * db

        if math.isclose(cost, cost_previous, rel_tol= 1e-20):
            break
        cost_previous = cost

        print(f'm: {m_curr}, b: {b_curr}, cost: {cost}, iteration: {i}')

    return m_curr, b_curr


if __name__ == '__main__':
    data = pd.read_csv('test_scores.csv')
    x = np.array(data.math)
    y = np.array(data.cs)

    gd_m, gd_b = gradient_descent(x, y)
    print(f'Using Gradient Descent \n m: {gd_m}, b: {gd_b}')

    m, b = using_sklearn()
    print(f'Using Sklearn prediction \n m: {m}, b: {b}')

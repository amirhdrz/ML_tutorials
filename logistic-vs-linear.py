#!/usr/bin/env python

"""Interactive program for demonstrating the differences between Linear Regression and Logistic Regression.

This script fits a linear regression model and a logistic regression model to the provided data
to solve the classification task.
The idea is to show the sensitivity of linear regression models to outliers. The user can
play with the figure by adding data points belonging to one class or the other, and seeing
how both models react.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from sklearn.linear_model import LinearRegression, LogisticRegression

__author__ = "Amir Hossein Heidari Zadi"
__email__ = "amir.hdrz@gmail.com"


def calculate_y(w, w0):
    """
    Given weight w and bias w0, calculates the y-coordinates from x.
    The weight and bias are assumed to be part of the following equation:
        w * (x,y) + w0 = 0
    """
    # We use squeeze() since w might be of the form array([[a, b]])
    # i.e. a single-dimensional array containing array of interest.
    w = w.squeeze()
    w0 = w0.squeeze()
    return ((-w[0] * x_lin) - w0) / w[1]  # Computes y given the weights and the bias.


def add_points(n, cls):
    """
    Using matplotlib ginput() function adds n points to the dataset
    belonging to class cls.
    :param n: Number of data points to add. Passed to ginput() function
    :param cls: either +1 or -1, referring to the positive class or the negative class.
    """
    global df
    if n > 0:
        pts = fig.ginput(n)  # Gets datapoints clicked by the user
        pts = pd.DataFrame(pts)
        pts[2] = pd.Series(cls, index=pts.index)  # Adds class label to the datapoints

        df = pd.concat([df, pts], ignore_index=True)  # Concatenates old data with the new data points

        # Re-fitting the models to the new data
        fit_models()

        # Updating the plots
        line1.set_ydata(calculate_y(linear.coef_, linear.intercept_))
        line2.set_ydata(calculate_y(logistic.coef_, logistic.intercept_))

        cls_p = df[df[2] > 0]  # Data points in the 'positive' class
        cls_n = df[df[2] < 0]  # Data points in the 'negative' class
        scatter1.set_offsets(cls_p.loc[:, 0:1])
        scatter2.set_offsets(cls_n.loc[:, 0:1])

        plt.draw()


def fit_models():
    """
    Fits the linear models, based on the DataFrame df
    """
    x = df.loc[:, 0:1]
    y = df.loc[:, 2]
    linear.fit(x, y)
    logistic.fit(x, y)


# Reads training data from csv file
df = pd.read_csv('test_data.csv', header=-1)

# Prepares the figure for plots
fig = plt.figure()
fig.add_subplot(111)
fig.subplots_adjust(bottom=0.25)

# Linear models to be fitted against the data
linear = LinearRegression()
logistic = LogisticRegression()

fit_models()

# Domain for plotting the linear models
x_lin = np.linspace(0, 0.8, num=100)

# Plots
line1, = plt.plot(x_lin, calculate_y(linear.coef_, linear.intercept_), label='linear')
line2, = plt.plot(x_lin, calculate_y(logistic.coef_, logistic.intercept_), label='logistic')

cls_p = df[df[2] > 0]  # Data points in the 'positive' class
cls_n = df[df[2] < 0]  # Data points in the 'negative' class

# Plots the individual data points
scatter1 = plt.scatter(cls_p.loc[:, 0], cls_p.loc[:, 1], marker='o', label='+1')
scatter2 = plt.scatter(cls_n.loc[:, 0], cls_n.loc[:, 1], marker='*', label='-1')
plt.legend()

# Adds buttons to the figure
btn_p = Button(plt.axes([0.25, 0.05, 0.1, 0.075]), 'Add +1')
btn_n = Button(plt.axes([0.40, 0.05, 0.1, 0.075]), 'Add -1')

# You can adjust number of points that are added when the buttons are clicked
btn_p.on_clicked(lambda event: add_points(5, 1))
btn_n.on_clicked(lambda event: add_points(5, -1))

plt.show()

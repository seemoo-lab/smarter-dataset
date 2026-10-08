#!/usr/bin/env python3

"""
Smartphone-based Communication Networks for
Emergency Response (smarter) Dataset
Copyright (C) 2018  Flor Alvarez
Copyright (C) 2018  Lars Almon
Copyright (C) 2018  Yannick Dylla
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import os
import sys
import re
import datetime
import logging
import warnings
from collections import Counter

import numpy

import config
import util

import scipy.stats as stats
import matplotlib.pyplot as plt

from matplotlib.ticker import FuncFormatter
import matplotlib.dates as mdates

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from plotly.offline import plot
    import plotly.graph_objs as go

# ECDF CODE FROM https://github.com/QuantEcon/QuantEcon.py 

class smarter_ecdf:
    """
    One-dimensional empirical distribution function given a vector of
    observations.

    Parameters
    ----------
    observations : array_like
        An array of observations

    Attributes
    ----------
    observations : array_like
        An array of observations

    """

    def __init__(self, observations):
        self.observations = numpy.asarray(observations)

    def __call__(self, x):
        """
        Evaluates the ecdf at x

        Parameters
        ----------
        x : scalar(float)
            The x at which the ecdf is evaluated

        Returns
        -------
        scalar(float)
            Fraction of the sample less than x

        """
        return numpy.mean(self.observations <= x)

    def plot(self, plots_dir, file_name, auto_open, traces, title = None, layout=None):

        if layout is None:
            if title is None: 
                title = ""
            layout = go.Layout(title=title, yaxis = dict(title = "Empirical CDF"))

        #sns.set()
        figure = go.Figure(data = go.Data([traces]), layout = layout)

        plot(figure, filename=os.path.join(plots_dir,file_name), show_link=False, auto_open=auto_open)

    def traces(self, a=None, b=None, xaxis = 'x', yaxis = 'y', name = '', isCoord = False, num = 200):
        """
        Plot the ecdf on the interval [a, b].

        Parameters
        ----------
        a : scalar(float), optional(default=None)
            Lower end point of the plot interval
        b : scalar(float), optional(default=None)
            Upper end point of the plot interval

        """

        # === choose reasonable interval if [a, b] not specified === #
        if a is None:
            a = self.observations.min() - self.observations.std()
        if b is None:
            b = self.observations.max() + self.observations.std()

        if a <= 0:
            a = self.observations.min()

        ### Used 2000 for log plot and 200 for others#
        x_vals, y_vals = self.coords(a,b, num)
        traces = go.Scatter(x = x_vals, y = y_vals, xaxis = xaxis, yaxis = yaxis)
        if isCoord:
            return self.coords(a,b, num)
        else:
            return traces

    def coords(self, a,b, num):
        x_vals = numpy.linspace(a, b, num)
        f = numpy.vectorize(self.__call__)
        y_vals = f(x_vals)

        return x_vals, y_vals
        
class smarter_histogram:
    """
    One-dimensional empirical distribution function given a vector of
    observations.

    Parameters
    ----------
    observations : array_like
        An array of observations

    Attributes
    ----------
    observations : array_like
        An array of observations

    """

    def __init__(self, observations):
        self.observations = numpy.asarray(observations)

    def plot(self, plots_dir, file_name, auto_open, traces, title = None, layout=None):
        if layout is None:
            if title is None: 
                title = "No title"
            layout = go.Layout(title=title, yaxis = dict(title = "Histogram"))

        # using numpy for histogram hist, bins = numpy.histogram(self.observations, bins = 10)

        #sns.set()

        figure = go.Figure(data = go.Data([traces]), layout = layout)

        plot(figure, filename=os.path.join(plots_dir,file_name), show_link=False, auto_open=auto_open)

    def traces(self, bin_size, a=None, b=None, norm = None, xaxis = 'x', yaxis = 'y', nan = None):
        """
        Plot the ecdf on the interval [a, b].

        Parameters
        ----------
        a : scalar(float), optional(default=None)
            Lower end point of the plot interval
        b : scalar(float), optional(default=None)
            Upper end point of the plot interval

        """

        # === choose reasonable interval if [a, b] not specified === #
        if nan is None:
            if a is None:
                a = int(min(self.observations)- numpy.std(self.observations))
            if b is None:
                b = int(max(self.observations) + numpy.std(self.observations))
        else:
            if a is None:
                a = int(min(self.observations)- numpy.nanstd(self.observations))
            if b is None:
                b = int(max(self.observations) + numpy.nanstd(self.observations))

        if a <= 0:
            a = int(min(self.observations))

        # using numpy for histogram hist, bins = numpy.histogram(self.observations, bins = 10)

        #sns.set()

        #color 30,30,80
        traces = go.Histogram(x = self.observations, 
            histnorm = norm, 
            autobinx = False, 
            xaxis = xaxis,
            yaxis = yaxis,
            xbins = dict(start=a, end = b, size = bin_size), 
            marker = dict(color = "rgb(40,90,130)", line = dict(width = 1, color = "rgb(0,0,0)")))

        return traces

def calculateMeanMedianVarStd(data, nan = None):
    dataAs = numpy.array(data)
    if nan is None:
        mean = numpy.mean(dataAs)
        median = numpy.median(dataAs)
        maximal = numpy.max(dataAs)
        std = numpy.std(dataAs)
    else:
        mean = numpy.nanmean(dataAs)
        median = numpy.nanmedian(dataAs)
        maximal = numpy.nanmax(dataAs)
        std = numpy.nanstd(dataAs)

    stats = [mean, median, maximal, std] 

    strResult = ("<b> mean= %.2f" %mean) +  ("<br> median= %.2f" %median) + ("<br> max= %.2f" %maximal) +  ("<br> std= %.2f" %std + "<b>") 

    return strResult, stats

def layoutConfig(title, axesX, axesY, bargap = 0.1, bargroupgap = 0.1, barmode = 'overlay', subtitle = None, 
                typeX = None, typeY = None, xaxis = None, yaxis = None, xaxis2 = None, yaxis2 = None, annotations = None):

    if subtitle is None: 
        subtitle = ""

    if typeX is None:
        typeX = "-"

    if typeY is None:
        typeY = "-"

    if annotations is None:
        annotations = []

    title = "<b>" + title + "</b>" + subtitle 

    if xaxis is None and yaxis is None:
        xaxis=dict(title=axesX, 
                    titlefont=dict(size=30, family = 'sans serif'), 
                    tickfont=dict(size=30, family = 'sans serif'), 
                    type = typeX, 
                    rangemode = 'tozero', 
                    showgrid = True, 
                    gridcolor='rgb(242,242,242)',
                    gridwidth=0.05, 
                    mirror='ticks',
                    showline = True, 
                    linecolor = 'rgb(0,0,0)',
                    linewidth = 2,
                    zerolinecolor='rgb(0, 0, 0)',
                    zerolinewidth=2)
        yaxis=dict(title=axesY, 
                    titlefont=dict(size=30, family = 'sans serif'), 
                    tickfont=dict(size=30, family = 'sans serif'), 
                    type = typeY, 
                    rangemode = 'nonnegative',
                    showgrid = True, 
                    gridcolor='rgb(242,242,242)',
                    gridwidth=0.05, 
                    mirror='ticks',
                    showline = True, 
                    linecolor = 'rgb(0,0,0)',
                    linewidth = 2,
                    zerolinecolor='rgb(0, 0, 0)',
                    zerolinewidth=2)

    if xaxis2 is None and yaxis2 is None:
        layout = go.Layout(
            title=title, 
            titlefont = dict(size=27), 
            xaxis=xaxis,
            yaxis=yaxis,
            bargap=bargap, 
            bargroupgap = bargroupgap, 
            barmode = barmode,
            annotations = annotations,
            legend=dict(font=dict(size=24)))

    else:
        layout = go.Layout(
            title=title, 
            titlefont = dict(size=27), 
            xaxis=xaxis,
            yaxis=yaxis,
            xaxis2=xaxis2,
            yaxis2=yaxis2,
            bargap=bargap, 
            bargroupgap = bargroupgap,
            barmode = barmode,
            annotations = annotations, 
            legend=dict(font=dict(size=24)))        

    return layout

def plotBoxPlot(x_data, y_data, title, axesX, axesY, boxpoints = ['all', 'outliers'], xaxis = 'x', yaxis = 'y', typeX = None, typeY = None,
                domainX = [0,1], anchorX = 'free', domainY = [0,1], anchorY = 'free', orientation = None, annotations = None, sourcedata = None):    
    traces = []
    #sns.set()

    if annotations is None:
        annotations = []

    if typeX is None:
        typeX = "-"

    if typeY is None:
        typeY = "-"

    if sourcedata is None:

        for x,y in zip(x_data, y_data):
            if orientation is None:
                traces.append(go.Box(
                    y=y,
                    x = x,
                    name=x,
                    boxpoints='all',
                    xaxis = xaxis,
                    yaxis = yaxis,
                    line=dict(width=1)
                ))
            else: 
                traces.append(go.Box(
                    x=y,
                    y = x,
                    name=x,
                    orientation = orientation,
                    boxpoints='all',
                    xaxis = xaxis,
                    yaxis = yaxis,
                    line=dict(width=1)
                ))
    else:
        traces = sourcedata

    xaxisdict = dict(titlefont=dict(size=26, family = 'sans serif'), 
                tickfont  = dict(size=26, family = 'sans serif'),
                rangemode = 'tozero', 
                showgrid = True, 
                gridcolor='rgb(242,242,242)',
                gridwidth=0.05, 
                showline = True, 
                zerolinecolor='rgb(0, 0, 0)', 
                zerolinewidth=2, 
                linecolor = 'rgb(0,0,0)',
                linewidth = 2)
    yaxisdict = dict(titlefont=dict(size=26, family = 'sans serif'), 
                tickfont  = dict(size=26, family = 'sans serif'),
                autorange=True,
                showgrid=True,
                zeroline=True,
                dtick=5,
                gridcolor='rgb(242,242,242)',
                gridwidth=0.05,
                zerolinecolor='rgb(0, 0, 0)',
                zerolinewidth=2, 
                rangemode = 'nonnegative', 
                showline = True, 
                linecolor = 'rgb(0,0,0)',
                linewidth = 2)

    if xaxis == 'x' and yaxis == 'y':
        layout = layoutConfig(title, axesX, axesY, xaxis = xaxisdict, yaxis = yaxisdict, annotations = annotations, typeX = typeX, typeY = typeY)

    else:
        xaxis2dict = dict(domain = domainX, anchor = anchorX, titlefont=dict(size=26, family = 'sans serif'), 
                        tickfont  = dict(size=26, family = 'sans serif'))
        yaxis2dict = dict(domain = domainY, anchor = anchorY, titlefont=dict(size=26, family = 'sans serif'), 
                tickfont  = dict(size=26, family = 'sans serif'))
        layout = layoutConfig(title, axesX, axesY, xaxis2 = xaxis2dict, yaxis2 = yaxis2dict, annotations = annotations, typeX = typeX, typeY = typeY)

    return [traces, layout]

def getMarker(data):
    array = [int(data * 0.1), int(data *0.3), int(data*0.5), 
             int(data * 0.7), int(data*0.9), int((data*1.0) - 1)]

    return array

def ecdfNeighborDistance(data): 
    ecdf1 = smarter_ecdf(data[0])
    ecdf2 = smarter_ecdf(data[1])
    ecdf3 = smarter_ecdf(data[2])


    x1,y1 = ecdf1.traces(name = "ECDF neighbors", isCoord = True)
    x2,y2 = ecdf2.traces(name = "ECDF neighbors", isCoord = True)
    x3,y3 = ecdf3.traces(name = "ECDF neighbors", isCoord = True)

    figure = plt.figure(1)
    ax = figure.add_subplot(111)
    ax.grid(True, linewidth=0.15)
    plots = []
    labels = ['d = 25 m', 'd = 44m', 'd = 110m']
    plots.append(ax.plot(x1, y1, '^', ls = '-', color='red', lw=0.75, markevery = getMarker(len(y1)))[0])
    plots.append(ax.plot(x2, y2, 's', ls = '-', color='blue', lw=0.75, markevery = getMarker(len(y2)))[0])
    plots.append(ax.plot(x3, y3, 'h', ls = '-', color='orange', lw=0.75, markevery = getMarker(len(y3)))[0])
    
    ax.set_xlabel('# of neighbors', fontdict = dict(fontweight="bold", fontname = 'serif', fontsize = 16))
    ax.set_ylabel('ECDF P(n<n1)', fontdict = dict(fontweight="bold", fontname = 'serif', fontsize = 16))
    ax.set_xlim(0,18)
    ax.set_ylim(0)
    for tick in ax.get_xticklabels():
        tick.set_fontsize(12)
    for tick in ax.get_yticklabels():
        tick.set_fontsize(12)

    ax.legend(plots, labels, loc='upper left')
    plt.show()

def ecdfUsingMatplot(data, xlim, xlabel, ylabel, num, isLog = False): 
    ecdf1 = smarter_ecdf(data[0])

    x1,y1 = ecdf1.traces(name = "ECDF neighbors", isCoord = True, num = num)

#    print("#### Plot DATA ###")
#    print("x1: %s" % x1)
#    print("y1: %s" % y1)
#    print ("#### END Plot DATA ###")
    figure = plt.figure(1)
    ax = figure.add_subplot(111)
    ax.grid(True, linewidth=0.15)
    plots = []
    if not isLog:
        plots.append(ax.plot(x1, y1, ls = '-')[0])
    else:
        plots.append(ax.semilogx(x1,y1, base = 10))
    ax.set_xlabel(xlabel, fontdict = dict(fontweight="bold", fontname = 'serif', fontsize = 16))
    ax.set_ylabel(ylabel, fontdict = dict(fontweight="bold", fontname = 'serif', fontsize = 16))
    ax.set_xlim(0,xlim)
    ax.set_ylim(0)
    
    for tick in ax.get_xticklabels():
        tick.set_fontsize(12)
    for tick in ax.get_yticklabels():
        tick.set_fontsize(12)

    plt.show()

def timeUsingMatplot(dataX, dataY, xlabel, ylabel): 
    figure = plt.figure(1)
    ax = figure.add_subplot(111)
    ax.grid(True, linewidth=0.15)
    plots = []
    labels = ['d = 25 m', 'd = 44m', 'd = 110m']
    x1 = dataX[0]
    x2 = dataX[1]
    x3 = dataX[2]
    y1 = dataY[0]
    y2 = dataY[1]
    y3 = dataY[2]

    len1 = len(x1) 
    len2 = len(x2)
    len3 = len(x3)  

    plots.append(ax.plot(x1, y1, '^', ls = '-', color='red', lw=0.5, markevery = [int(len1 * 0.1), int(len1*0.3), int(len1*0.5), int(len1*0.7), int(len1*0.9)])[0])
    plots.append(ax.plot(x2, y2, 's', ls = '-', color='blue', lw=0.5, markevery = [int(len1 * 0.2), int(len1*0.4), int(len1*0.6), int(len1*0.8), int(len1*0.9)])[0])
    plots.append(ax.plot(x3, y3, 'h', ls = '-', color='orange', lw=0.5, markevery = [int(len1 * 0.1), int(len1*0.3), int(len1*0.5), int(len1*0.7), int(len1*0.9)])[0])

    ax.set_xlabel(xlabel, fontdict = dict(fontweight="bold", fontname = 'serif', fontsize = 16))
    ax.set_ylabel(ylabel, fontdict = dict(fontweight="bold", fontname = 'serif', fontsize = 16))

    ax.set_ylim(0)
    plt.gcf().autofmt_xdate()
    formatDate = mdates.DateFormatter('%H:%M')
    plt.gca().xaxis.set_major_formatter(formatDate)
    
    for tick in ax.get_xticklabels():
        tick.set_fontsize(12)
    for tick in ax.get_yticklabels():
        tick.set_fontsize(12)

    ax.legend(plots, labels, loc='upper left')
    plt.show()

def timeUsingMatplotUnique(dataX, dataY, xlabel, ylabel): 
    figure = plt.figure(1)
    ax = figure.add_subplot(111)
    ax.grid(True, linewidth=0.15)
    plots = []

    if hasattr(dataX[0], '__len__'):
        x1 = dataX[0]
        y1 = dataY[0]
    else:
        x1 = dataX
        y1 = dataY

    len1 = len(x1) 

    plots.append(ax.plot(x1, y1, ls = '-', lw=0.75))#, markevery = [int(len1 * 0.1), int(len1*0.3), int(len1*0.5), int(len1*0.7), int(len1*0.9)])[0])
 
    ax.set_xlabel(xlabel, fontdict = dict(fontweight="bold", fontname = 'serif', fontsize = 16))
    ax.set_ylabel(ylabel, fontdict = dict(fontweight="bold", fontname = 'serif', fontsize = 16))

    ax.set_ylim(0)
    plt.gcf().autofmt_xdate()
    formatDate = mdates.DateFormatter('%H:%M')
    plt.gca().xaxis.set_major_formatter(formatDate)
    
    for tick in ax.get_xticklabels():
        tick.set_fontsize(12)
    for tick in ax.get_yticklabels():
        tick.set_fontsize(12)

    #ax.legend(plots, loc='upper left')
    plt.show()

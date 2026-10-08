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

###################################
# START: including latex support for fonts
###################################
import matplotlib 
# as mpl
matplotlib.use("pgf")
pgf_with_pdflatex = {
    "pgf.texsystem": "pdflatex",
    "font.family": "serif", # use serif/main font for text elements
    "text.usetex": True,    # use inline math for ticks
    "axes.labelsize" : 11,
    # "text.fontsize"  : 11,
    "legend.fontsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,  
    "figure.figsize": [4.7, 3.33], 
    "figure.dpi" : 80,
    "savefig.dpi" : 300,
    "pgf.preamble": [
         r"\usepackage[utf8x]{inputenc}",
         r"\usepackage[T1]{fontenc}",
         r"\usepackage[osf]{mathpazo}",
         ]
}
matplotlib.rcParams.update(pgf_with_pdflatex)

###################################
# END
###################################

import matplotlib.pyplot as plt

import pandas as pd
from math import pi

from matplotlib.ticker import FuncFormatter
import matplotlib.dates as mdates

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from plotly.offline import plot
    import plotly.graph_objs as go

from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D

import scipy.stats as ss

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
        
def getMarker(data):
    array = [int(data * 0.1), int(data *0.3), int(data*0.5), 
             int(data * 0.7), int(data*0.9), int((data*1.0) - 1)]

    return array

def ecdfNeighborDistance(data,xlabel, ylabel, title): 
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
    plots.append(ax.plot(x1, y1, '^', ls = '-', color=config.FIRST,   lw=config.LW, markersize=3, markeredgewidth=0.15, markevery = getMarker(len(y1)))[0])
    plots.append(ax.plot(x2, y2, 's', ls = '-', color=config.SECOND,  lw=config.LW, markersize=3, markeredgewidth=0.15, markevery = getMarker(len(y2)))[0])
    plots.append(ax.plot(x3, y3, 'h', ls = '-', color=config.THIRD,   lw=config.LW, markersize=3, markeredgewidth=0.15, markevery = getMarker(len(y3)))[0])
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(0,18)
    ax.set_ylim(0)

    ax.set_yticks([0.2,0.4,0.6, 0.8, 1.0])
    plt.xticks([2,4,6, 8, 10, 12, 14, 16, 18], ["2","4","6", "8", "10", "12", "14", "16",  "18"])

    ax.legend(plots, labels, loc='upper center', bbox_to_anchor=(0.5,1.225), fancybox=True, shadow=False, ncol=5)
    figure.savefig(config.PLOTS_CHANTS_DISS_DIR + title, bbox_inches='tight', dpi = 300)

def ecdfNeighborDegreeAxes(dataN, dataD, xlabel, xlabel2, ylabel, title): 
    ecdf1 = smarter_ecdf(dataN[0])
    ecdf2 = smarter_ecdf(dataN[1])
    ecdf3 = smarter_ecdf(dataN[2])
    ecdf4 = smarter_ecdf(dataD)

    x1,y1 = ecdf1.traces(name = "ECDF neighbors", isCoord = True)
    x2,y2 = ecdf2.traces(name = "ECDF neighbors", isCoord = True)
    x3,y3 = ecdf3.traces(name = "ECDF neighbors", isCoord = True)
    x4,y4 = ecdf4.traces(name = "ECDF neighbors", isCoord = True)

    figure, axs = plt.subplots(2, sharex = True, sharey = True)
    # ax = figure.add_subplot(211)
    axs[0].grid(True, linewidth=0.15)
    axs[1].grid(True, linewidth=0.15)
    plots1 = []
    plots0 = []
    labels1 = ['d = 25 m', 'd = 44m','d = 110m']
    labels0 = ['Node degree']
    plots1.append(axs[1].plot(x1, y1, '^', ls = '-', color=config.FIRST,   lw=config.LW, markersize=3, markeredgewidth=0.15, markevery = getMarker(len(y1)))[0])
    plots1.append(axs[1].plot(x2, y2, 's', ls = '-', color=config.SECOND,  lw=config.LW, markersize=3, markeredgewidth=0.15, markevery = getMarker(len(y2)))[0])
    plots1.append(axs[1].plot(x3, y3, 'h', ls = '-', color=config.THIRD,   lw=config.LW, markersize=3, markeredgewidth=0.15, markevery = getMarker(len(y3)))[0])
    plots0.append(axs[0].plot(x4, y4, ls = '-', color=config.FOUR,    lw=config.LW)[0])
    
    axs[1].set_xlabel(xlabel)
    axs[1].set_ylabel(ylabel)


    axs[0].set_xlabel(xlabel2)
    axs[0].set_ylabel(ylabel)

    axs[1].set_xlim(0,18)
    axs[1].set_ylim(0)

    plt.yticks([0.2,0.4,0.6, 0.8, 1.0], ["0.2","0.4", "0.6", "0.8", "1.0"])
    plt.xticks([2,4,6, 8, 10, 12, 14, 16, 18], ["2","4","6", "8", "10", "12", "14", "16",  "18"])

    axs[1].legend(plots1, labels1, loc='best', fancybox=True, shadow=False, ncol=1)
    figure.savefig(config.PLOTS_CHANTS_DISS_DIR + title, bbox_inches='tight', dpi = 300)

def ecdfUsingMatplot(data, xlim, xlabel, ylabel, num, title, isLog = False): 
    ecdf1 = smarter_ecdf(data[0])

    x1,y1 = ecdf1.traces(name = "ECDF neighbors", isCoord = True, num = num)

    figure = plt.figure(1)
    ax = figure.add_subplot(111)
    ax.grid(True, linewidth=0.15)
    plots = []
    if not isLog:
        plots.append(ax.plot(x1, y1, ls = '-', color = config.FIRST, lw=config.LW)[0])
    else:
        plots.append(ax.semilogx(x1,y1, basex = 10, ls = '-', color = config.FIRST, lw=config.LW))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(0,xlim)
    ax.set_ylim(0)
    ax.set_yticks([0.2,0.4,0.6, 0.8, 1.0])
    
    figure.savefig(config.PLOTS_CHANTS_DISS_DIR + title, bbox_inches='tight', dpi = 300)

def timeUsingMatplot(dataX, dataY, xlabel, ylabel, title): 
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

    ax.set_yticks([3,6,9,12,15,18])

    plots.append(ax.plot(x1, y1, '^', ls = '-',lw=config.LW, color=config.FIRST, markevery = [int(len1 * 0.1),int(len1 * 0.15), int(len1*0.3), int(len1 * 0.35),int(len1*0.5),int(len1 * 0.55), int(len1*0.7), int(len1 * 0.75),int(len1*0.9), int(len1 * 0.95)], markersize=3, markeredgewidth=0.15)[0])
    plots.append(ax.plot(x2, y2, 's', ls = '-',lw=config.LW, color=config.SECOND, markevery = [int(len1 * 0.2),int(len1 * 0.25), int(len1*0.4), int(len1 * 0.45),int(len1*0.6), int(len1 * 0.65),int(len1*0.8), int(len1 * 0.85),int(len1*0.9), int(len1 * 0.95)], markersize=3, markeredgewidth=0.15)[0])
    plots.append(ax.plot(x3, y3, 'h', ls = '-',lw=config.LW, color=config.THIRD, markevery = [int(len1 * 0.1),int(len1 * 0.15), int(len1*0.3), int(len1 * 0.35),int(len1*0.5),int(len1 * 0.55), int(len1*0.7), int(len1 * 0.75),int(len1*0.9), int(len1 * 0.95)], markersize=3, markeredgewidth=0.15)[0])

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    ax.set_ylim(0)
    plt.gcf().autofmt_xdate()
    formatDate = mdates.DateFormatter('%H:%M')
    plt.gca().xaxis.set_major_formatter(formatDate)

    ax.legend(plots, labels, loc='upper center', bbox_to_anchor=(0.5,1.315), fancybox=True, shadow=False, ncol=5)
        
    figure.savefig(config.PLOTS_CHANTS_DISS_DIR+ title, bbox_inches='tight', dpi = 300)

def timeDegreeNeighbor(Xdegree, Ydegree, Xn, Yn, xlabel, ylabel, title): 
    figure = plt.figure(1)
    ax = figure.add_subplot(111)
    ax.grid(True, linewidth=0.15)
    plots = []
    labels = ['Node degree', '\# of neighbours (d = 110m)']
    x2 = Xn[2]
    
    y2 = Yn[2]

    if hasattr(Xdegree[0], '__len__'):
        x1 = Xdegree[0]
        y1 = Ydegree[0]
    else:
        x1 = Xdegree
        y1 = Ydegree
    
    len1 = len(x1) 
    len2 = len(x2)

    ax.set_yticks([3,6,9,12,15,18])

    plots.append(ax.plot(x1, y1, '^', ls = '-',lw=config.LW, color=config.FOUR, markevery = [int(len1 * 0.1), int(len1*0.3), int(len1*0.5), int(len1*0.7), int(len1*0.9)], markersize=3, markeredgewidth=0.15)[0])
    plots.append(ax.plot(x2, y2, 's', ls = '-',lw=config.LW, color=config.THIRD, markevery = [int(len1 * 0.2), int(len1*0.4), int(len1*0.6), int(len1*0.8), int(len1*0.9)], markersize=3, markeredgewidth=0.15)[0])

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    ax.set_ylim(0)
    plt.gcf().autofmt_xdate()
    formatDate = mdates.DateFormatter('%H:%M')
    plt.gca().xaxis.set_major_formatter(formatDate)

    ax.legend(plots, labels, loc='upper center', bbox_to_anchor=(0.5,1.3), fancybox=True, shadow=False, ncol=5)
        
    figure.savefig(config.PLOTS_CHANTS_DISS_DIR+ title, bbox_inches='tight', dpi = 300)

def timeUsingMatplotUnique(dataX, dataY, xlabel, ylabel, title): 
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

    plots.append(ax.plot(x1, y1, ls = '-', lw=config.LW, color = config.FIRST))#, markevery = [int(len1 * 0.1), int(len1*0.3), int(len1*0.5), int(len1*0.7), int(len1*0.9)])[0])
 
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    ax.set_ylim(0)
    # for number of connections
    # ax.set_yticks([1,2,3,4,5,6,7,8])
    # ax.set_yticks([3,6,9,12,15])

    # for multiuse cluster coeficient
    # ax.set_yticks([0.1,0.2,0.3,0.4,0.5])

    # for number of messages
    # ax.set_yticks([,2,3,4,5,6,7,8])

    ax.set_yticks([150,300,450,600,750])

    plt.gcf().autofmt_xdate()
    formatDate = mdates.DateFormatter('%H:%M')
    plt.gca().xaxis.set_major_formatter(formatDate)

    # for message count hops
    # ax.set_yticks([15, 30, 45, 60, 75, 90])

    #ax.legend(plots, loc='upper left')
    figure.savefig(config.PLOTS_CHANTS_DISS_DIR + title, bbox_inches='tight', dpi = 300)

def messageDelay(bestX, bestY, meanX, meanY, xlabel, ylabel, title): 
    figure = plt.figure(1)
    ax = figure.add_subplot(111)
    ax.grid(True, linewidth=0.15)
    plots = []

    if hasattr(bestX[0], '__len__'):
        x1 = bestX[0]
        y1 = bestY[0]
    else:
        x1 = bestX
        y1 = bestY

    if hasattr(bestX[0], '__len__'):
        x2 = meanX[0]
        y2 = meanY[0]
    else:
        x2 = meanX
        y2 = meanY

    len1 = len(x1) 
    len2 = len(x2)

    plots.append(ax.plot(x1, y1, '^', ls = '-', lw=config.LW, color = config.FIRST, markersize=3, markeredgewidth=0.15, markevery = [int(len1 * 0.1), int(len1*0.3), int(len1*0.5), int(len1*0.7), int(len1*0.9)])[0])
    plots.append(ax.plot(x2, y2, 's', ls = '-', lw=config.LW, color = config.SECOND,markersize=3, markeredgewidth=0.15,  markevery = [int(len2 * 0.1), int(len2*0.3), int(len2*0.5), int(len2*0.7), int(len2*0.9)])[0])
 
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    ax.set_xlim(0,65)
    ax.set_ylim(0)
    # for number of connections
    # ax.set_yticks([1,2,3,4,5,6,7,8])
    # ax.set_yticks([3,6,9,12,15])

    # for multiuse cluster coeficient
    # ax.set_yticks([0.1,0.2,0.3,0.4,0.5])

    # for number of messages
    # ax.set_yticks([,2,3,4,5,6,7,8])
    # ax.set_yticks([150,300,450,600,750])

    # plt.gcf().autofmt_xdate()
    # formatDate = mdates.DateFormatter('%H:%M')
    # plt.gca().xaxis.set_major_formatter(formatDate)

    # for message count hops
    ax.set_yticks([15, 30, 45, 60, 75, 90])

    #ax.legend(plots, loc='upper left')

    labels = ['best multicast', 'median multicast']
    ax.legend(plots, labels, loc='upper center', bbox_to_anchor=(0.5,1.275), fancybox=True, shadow=False, ncol=3)
    figure.savefig(config.PLOTS_CHANTS_DISS_DIR + title, bbox_inches='tight', dpi = 300)

def timeAllMessages(dataX, dataY, xlabel, ylabel, title): 
    figure = plt.figure(1)
    ax = figure.add_subplot(111)
    ax.grid(True, linewidth=0.15)
    plots = []

    if hasattr(dataX[0], '__len__'):
        x1 = dataX[0]
        y1 = dataY[0]
        x2 = dataX[1]
        y2 = dataY[1]
        x3 = dataX[2]
        y3 = dataY[2]
        x4 = dataX[3]
        y4 = dataY[3]
        x5 = dataX[4]
        y5 = dataY[4]
        
    else:
        x1 = dataX
        y1 = dataY

    len1 = len(x1) 

    print(len(x1))

    plots.append(ax.plot(x1, y1, ls = '-', lw=config.LW, color = config.FIRST)[0])#, markevery = [int(len1 * 0.1), int(len1*0.3), int(len1*0.5), int(len1*0.7), int(len1*0.9)])[0])
    plots.append(ax.plot(x3, y3, ls = '-', lw=config.LW, color = config.THIRD)[0])#, markevery = [int(len1 * 0.1), int(len1*0.3), int(len1*0.5), int(len1*0.7), int(len1*0.9)])[0])
    plots.append(ax.plot(x4, y4, ls = '-', lw=config.LW, color = config.FOUR)[0])#, markevery = [int(len1 * 0.1), int(len1*0.3), int(len1*0.5), int(len1*0.7), int(len1*0.9)])[0])
    plots.append(ax.plot(x5, y5, ls = '-', lw=config.LW, color = config.FIVE)[0])#, markevery = [int(len1 * 0.1), int(len1*0.3), int(len1*0.5), int(len1*0.7), int(len1*0.9)])[0])
    plots.append(ax.plot(x2, y2, ls = '-', lw=config.LW, color = config.SECOND)[0])#, markevery = [int(len1 * 0.1), int(len1*0.3), int(len1*0.5), int(len1*0.7), int(len1*0.9)])[0])
 
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    ax.set_ylim(0)
    # for number of connections
    # ax.set_yticks([1,2,3,4,5,6,7,8])
    # ax.set_yticks([3,6,9,12,15])

    # for multiuse cluster coeficient
    # ax.set_yticks([0.1,0.2,0.3,0.4,0.5])

    # for number of messages
    # ax.set_yticks([,2,3,4,5,6,7,8])

    ax.set_yticks([10,20,30,40,50])

    plt.gcf().autofmt_xdate()
    formatDate = mdates.DateFormatter('%H:%M')
    plt.gca().xaxis.set_major_formatter(formatDate)

    #ax.legend(plots, loc='upper left')

    # message_types_en = {"hilferuf": "SOS Emergency Messages","personenfinder": "Person-Finder","lebenszeichen": "I am Alive Notification","ressourcenmarkt": "Resource Market Registry","chat": "Messaging Services"}

    labels = ['Messaging', 'I am Alive','Person-Finder', 'Resource Market', 'SOS']
    ax.legend(plots, labels, loc='upper center', bbox_to_anchor=(0.5,1.5), fancybox=True, shadow=False, ncol=3)
    figure.savefig(config.PLOTS_CHANTS_DISS_DIR + title, bbox_inches='tight', dpi = 300)
    # plt.show()

def radar_factory(num_vars, frame='circle'):
    """Create a radar chart with `num_vars` axes.

    This function creates a RadarAxes projection and registers it.

    Parameters
    ----------
    num_vars : int
        Number of variables for radar chart.
    frame : {'circle' | 'polygon'}
        Shape of frame surrounding axes.

    """
    # calculate evenly-spaced axis angles
    theta = numpy.linspace(0, 2*numpy.pi, num_vars, endpoint=False)

    class RadarAxes(PolarAxes):

        name = 'radar'
        # use 1 line segment to connect specified points
        RESOLUTION = 1

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # rotate plot such that the first axis is at the top
            self.set_theta_zero_location('N')

        def fill(self, *args, closed=True, **kwargs):
            """Override fill so that line is closed by default"""
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = numpy.concatenate((x, [x[0]]))
                y = numpy.concatenate((y, [y[0]]))
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(numpy.degrees(theta), labels)

        def _gen_axes_patch(self):
            # The Axes patch must be centered at (0.5, 0.5) and of radius 0.5
            # in axes coordinates.
            if frame == 'circle':
                return Circle((0.5, 0.5), 0.5)
            elif frame == 'polygon':
                return RegularPolygon((0.5, 0.5), num_vars,
                                      radius=.5, edgecolor="k")
            else:
                raise ValueError("unknown value for 'frame': %s" % frame)

        def _gen_axes_spines(self):
            if frame == 'circle':
                return super()._gen_axes_spines()
            elif frame == 'polygon':
                # spine_type must be 'left'/'right'/'top'/'bottom'/'circle'.
                spine = Spine(axes=self,
                              spine_type='circle',
                              path=Path.unit_regular_polygon(num_vars))
                # unit_regular_polygon gives a polygon of radius 1 centered at
                # (0, 0) but we want a polygon of radius 0.5 centered at (0.5,
                # 0.5) in axes coordinates.
                spine.set_transform(Affine2D().scale(.5).translate(.5, .5)
                                    + self.transAxes)
                return {'polar': spine}
            else:
                raise ValueError("unknown value for 'frame': %s" % frame)

    register_projection(RadarAxes)
    return theta

def radarUsingMatplot(data_unique, data_all, labels, colors, title):
    N = 5

    theta = radar_factory(N, frame = 'polygon')

    length_all = sum(data_all)
    length_unique = sum(data_unique)

    percentage_all = [round((i*100)/length_all, 2) for i in data_all]
    percentage_unique = [round((i*100)/length_unique,2) for i in data_unique]

    data = [labels, ('all', [percentage_unique, percentage_all])]

    plots = []
    spoke_labels = data.pop(0)

    figure, axes = plt.subplots(figsize=(N, N), nrows=1, ncols=1,
                             subplot_kw=dict(projection='radar'))
    figure.subplots_adjust(wspace=0.25, hspace=0.20, top=0.85, bottom=0.05)

    # colors = ['b', 'r', 'g', 'm', 'y']
    label = ['Services Usage', 'Network Usage']
    # Plot the four cases fromb the example data on separate axes
    for t, case_data in data:
        for d, color, l in zip(case_data, colors, label):
            logging.info(d)
            plots.append(axes.plot(theta, d, color=color, label = l))
            axes.fill(theta, d, facecolor=color, alpha=0.25)
        axes.set_varlabels(spoke_labels)
     
    # # Draw ylabels
    axes.set_rlabel_position(0)
    plt.yticks([10,20,30, 40, 50, 60, 70], ["10","20","30", "40", "50", "60", "70"], color="grey")
    plt.ylim(0,70)
     
    # # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 1.1))
    figure.savefig(config.PLOTS_CHANTS_DISS_DIR + title, bbox_inches='tight', dpi = 300)

def speedBreak(data, title):
    ######## START BREAK FIGURE ##########
    plots = []
    f, (ax, ax2) = plt.subplots(2, 1, sharex=True)

    ax.grid(True, linewidth=0.15)
    ax2.grid(True, linewidth=0.15)

    # The required parameters
    num_steps = 5#5
    max_percentage = 0.5 #0.5
    num_bins = 108#150

    # Calculating the maximum value on the y axis and the yticks
    max_val = max_percentage * len(data)
    step_size = max_val / num_steps
    yticks = [ x * step_size  for x in range(3, 4+1) ]
    yticks2 = [ x * step_size * 0.1 for x in range(0, 4+1) ]


    # Running the histogram method
    ax2.hist(data, num_bins, edgecolor = 'black', color = config.FIRST)
    ax.hist(data, num_bins, edgecolor = 'black', color = config.FIRST)

    # To plot correct percentages in the y axis     
    to_percentage = lambda y, pos: str(round( ( y / float(len(data)) ) * 100.0, 2)) 
    ax2.yaxis.set_major_formatter(FuncFormatter(to_percentage))
    ax.yaxis.set_major_formatter(FuncFormatter(to_percentage))

    ax2.set_xlabel('Speed in [m/s]')
    ax2.set_yticks( yticks2 )
    ax.set_yticks( yticks )
    ax2.set_ylim(0, max_val/18)
    ax.set_ylim(max_val/2, max_val)

    # hide the spines between ax and ax2
    ax.spines['bottom'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax.xaxis.tick_top()
    ax.tick_params(labeltop=False)  # don't put tick labels at the top
    ax2.xaxis.tick_bottom()
    d = .0025  # how big to make the diagonal lines in axes coordinates
    # arguments to pass to plot, just so we don't keep repeating them
    kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
    plots.append(ax.plot((-d, +d), (-d, +d), **kwargs))        # top-left diagonal
    plots.append(ax.plot((1 - d, 1 + d), (-d, +d), **kwargs))  # top-right diagonal

    kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
    plots.append(ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs))  # bottom-left diagonal
    plots.append(ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs))  # bottom-right diagonal

    ax.set_ylabel('')
    ax2.set_ylabel('% of time (in 10 [s] slot) \n')
    ax2.yaxis.set_label_coords(-0.075, 1.10)

    # save as PDF
    f.savefig(config.PLOTS_CHANTS_DISS_DIR + title, bbox_inches='tight', dpi = 300)
    ######## END BREAK FIGURE ##########

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


import logging
import sys
import os

start_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
script_name = os.path.basename(sys.argv[0]).replace(".py","")
log_dir = os.path.join(start_dir, "logs")

os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(level=logging.DEBUG, handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(os.path.join(log_dir, "%s.log" % script_name))], format="%(asctime)s %(levelname)s: %(message)s", datefmt="%d-%m-%Y %H:%M:%S")

POOL_SIZE = 12  # number of processes used for multiprocessing

DATA_PATH = '../data'  # path of raw data with ibr-logs, realm-db, sensor-traces dirs & contactlist.csv
PLOTS_DIR = '../plots'  # output folder for 
PYTHON_DIR = '../python/'  # output folder for plots

PLOTS_CHANTS_DISS_DIR = "../results"


DATA_SUB_DIR = "processed"  # DATA_PATH sub folder where the prepared data should be stored
DATA_DIR = os.path.join(DATA_PATH, DATA_SUB_DIR)  # shortcut
AUTO_OPEN_PLOTS = False  # flag indicating if plotly plots should open themself after creation

FROM_TO = ["2017-09-02 09:30:00","2017-09-02 16:30:00"]  # time range of the field experiment, used for filtering and aligning data
WALKING_FROM_TO = ["2017-09-02 10:30:00","2017-09-02 15:30:00"]  # time range of the field experiment in which all test persons where only walking

# prepare_data.py
MAX_ALT = 1000  # gps data with higher altitude is ignored (high altitude values are a good indicator for unpercise position, often during startup)

# smarter_movements.py
KALMAN_ITERATIONS = 5  # number iterations for the kalman filter
MAX_SMOOTH_GAP = 15*60  # max gps gap length in seconds which will be interpolated/closed when kalman smoothing is used. Set to None for all & 0 for none
PLOT_ALL_GPS_TRACKS = False  # set this to True if you want the gps track of each node in a single file, otherwise they are plotted all into one
GPS_SIMPLIFY = True  # simplifies gps tracks with ramer–douglas–peucker algorithm for plotting, recommend for performance reasons

MAX_NEIGHBOR_DISTANCE = 25  # max distance in which nodes count as neighbors
MAX_DISTANCE_ARRAY = [25, 44, 110]

# smarter_connections.py
CONN_LOOKUP_OFFSET = 1  # offset in seconds in both directions to determine which connection events belong to the same connection
GPS_LOOKUP_OFFSET = 1  # offset in seconds in both directions during gps postion lookup for connection distance calculation (mean)

MAX_CONN_DISTANCE = 200  # max allowed connection distance
MAX_END_GPS_GAP = 15*60  # max gps gap in seconds for wich the end/right side is used to close the connection if the distance after the gap is > MAX_CONN_DISTANCE

CONNECTIONS_COMPARE_REPORTS = None  # array of paths to The One ConnectivityONEReport files for which the clustering coefficient is calculated to compare it with the smarter trace. Set to None to disable comparison
CONNECTIONS_COMPARE_NAMES = None  # names of the compare report files (displayed in plots)

# smarter_messages.py
MESSAGE_MERGE_OFFSET = 10  # offset in seconds in both directions in which receive & delivered events are fused to one transferd event, if this catches multiple events an exception is thrown
MAX_TRANSFER_CORRECTION = 600  # max offset in seconds for wich a transferred event is moved, to match the logical sorting.
MAX_SEND_OFFSET = 999  # max offset in milliseconds in which a send event is created which starts the sending for a transferd event
MAX_USE_BEFORE_CREATE = 50  # max amount in seconds for which events that use a bundle before its create event are moved behind the create event. Events with larger offset are ignored
CREATE_GENERATION = 1  # generates create & transferred events to nodes which use the bundle x seconds after its first use. Set to None to delte bundles without create event.

PLOT_ALL_BUNDLE_PATHS = True  # set True to plot the path of all bundles, very slow ~2000 bundles

FIVE = '#d62728' # red
FIRST = '#8c564b' # brown
SECOND = '#217821' # green
THIRD = '#ff7f0e' # orange
FOUR = '#1f77b4' # blue

LW = 0.75

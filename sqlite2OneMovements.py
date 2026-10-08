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

import logging
import os
import sys
import sqlite3
import utm
import pandas
import argparse
import multiprocessing
from datetime import datetime
import re
from operator import itemgetter

from natsort import natsorted
# from natsort import natsorted, index_natsorted, order_by_index

START_DIR = os.path.dirname(os.path.realpath(__file__))
SCRIPT_NAME = os.path.splitext(os.path.basename(__file__))[0]
LOG_PATH = os.path.join(START_DIR, "%s.log" % SCRIPT_NAME)

logging.basicConfig(level=logging.DEBUG, handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(LOG_PATH)], format="%(asctime)s %(levelname)s: %(message)s", datefmt="%d-%m-%Y %H:%M:%S")


def func_wrapper(func, args):
    try:
        return func(*args)
    except KeyboardInterrupt:
        return


def forAll(input_dir, file_ending, worker, func, args=None):
    start = datetime.now()

    logging.debug("Starting %s ..." % func.__name__)

    counter = 0
    results = []
    processes = []

    pool = multiprocessing.Pool(processes=worker)

    for f in natsorted(os.listdir(input_dir)):
        absPath = os.path.abspath(os.path.join(input_dir,f))

        select = False
        if file_ending is None:
            if os.path.isdir(absPath):
                select = True
        else:
            if os.path.isfile(absPath) and absPath.endswith(file_ending):
                select = True

        if select:
            if args is None:
                func_args = (counter, absPath)
            else:
                func_args = (counter, absPath) + args
            counter += 1

            p = pool.apply_async(func=func_wrapper, args=(func,func_args))
            processes.append(p)

    for p in processes:
        result = p.get()
        if result is not None:
            results.append(result)

    pool.close()

    logging.debug("All %s finished after %.2f seconds" % (func.__name__, (datetime.now()-start).total_seconds()))

    return results


def parallelized_apply(dataframes, worker, func, args=None):
    start = datetime.now()

    processes = []

    pool = multiprocessing.Pool(processes=worker)

    for i, df in enumerate(dataframes):
        if args is None:
            func_args = (df,i)
        else:
            func_args = (df,i) + args

        p = pool.apply_async(func=func_wrapper, args=(func, func_args))
        processes.append(p)

    results = []
    for p in processes:
        try:
            result = p.get()
        except KeyboardInterrupt:
            logging.info("Got ^C terminating...")
            pool.terminate()
            sys.exit(-1)

        if result is not None:
            results += result

    pool.close()

    logging.debug("All %s finished after %.2f seconds" % (func.__name__, (datetime.now()-start).total_seconds()))

    return results


def createOneMovementsWorker(counter,db_path,table_name,lat_column,lon_column,time_column,rename,debug):
    conn = sqlite3.connect(db_path)

    try:
        coords = pandas.read_sql("SELECT %s AS lat, %s AS lon, %s AS t FROM %s" % (lat_column, lon_column, time_column, table_name), conn, index_col="t", parse_dates=["t"])
        conn.close()
    except pandas.io.sql.DatabaseError as e:
        conn.close()
        coords = []
        if "no such table" in str(e):
            logging.error("Database %s has no table called '%s'" % (os.path.basename(db_path), table_name))
        else:
            raise e


    # if no gps data only pass id to create a 0,0 point in createOneMovementsScaler. Needed because the one otherwise reorders node ids
    if rename:
        _id = "%s" % counter
    else:
        _id = os.path.splitext(os.path.basename(db_path))[0]


    if len(coords) > 0:
        def transform_coords(row):
            return utm.from_latlon(row["lat"],row["lon"])
        utm_coords = coords.apply(transform_coords, axis=1)

        utm_coords = pandas.DataFrame(utm_coords.values.tolist(),index=coords.index, columns=["easting", "northing", "zone_number", "zone_letter"])

        zone_numbers = utm_coords["zone_number"].unique()
        zone_letters = utm_coords["zone_letter"].unique()

        if len(zone_numbers) > 1 or len(zone_letters) > 1:
            logging.error("Multiple UTM Zones detected! %s" % zone_numbers)
            raise NotImplementedError("Multiple UTM Zones detected! %s" % zone_numbers)

        zone = {"letter":zone_letters[0], "number": zone_numbers[0]}

        utm_coords.drop("zone_number", axis=1, inplace=True)
        utm_coords.drop("zone_letter", axis=1, inplace=True)

        # resample
        utm_coords = utm_coords.resample("1S").mean()

        if debug:
            # just on NaN per gap
            utm_coords = utm_coords[utm_coords.ffill(limit=1).notnull()["easting"]]
        else:
            # delete NaNs
            utm_coords.dropna(axis="index", how="any", inplace=True)

        min_t = utm_coords.index.min()
        max_t = utm_coords.index.max()

        min_x = utm_coords["easting"].min()
        max_x = utm_coords["easting"].max()

        min_y = utm_coords["northing"].min()
        max_y = utm_coords["northing"].max()

        utm_coords["id"] = _id

        return_data = (min_t,max_t,zone,min_x,max_x,min_y,max_y,utm_coords)

    else:
        logging.error("No data for %s" % os.path.basename(db_path))
        return_data = (None,None,None,None,None,None,None,_id)

    if rename:
        orig_filename = re.sub("^\d+_","", os.path.basename(db_path), count=1)
        orig_split = os.path.splitext(orig_filename)
        new_path = os.path.join(os.path.dirname(db_path),"%s_%s_%i%s" % (counter,orig_split[0], rename, orig_split[1]))
        os.rename(db_path, new_path)

    return return_data


def createOneMovementsScaler(utm_coords,counter,num_nodes,min_t,min_x,max_y,diff_x,diff_y,debug):
    # if no gps data exists utm_coords is the string id, need to set node 0,0 otherwise The One reorders the id mapping
    if isinstance(utm_coords, str):
        utm_coords = pandas.DataFrame([{"id":utm_coords,"x":0,"y":0}], index=[0], columns=["id","x","y"])
        utm_coords["time"] = utm_coords.index
        return utm_coords.to_dict("records")

    def transform_x(x):
        return x - min_x
    utm_coords["x"] = utm_coords["easting"].apply(transform_x)

    def transform_y(y):
        # the one has (0,0) at top left
        return max_y - y
    utm_coords["y"] = utm_coords["northing"].apply(transform_y)

    utm_coords.drop("easting", axis=1, inplace=True)
    utm_coords.drop("northing", axis=1, inplace=True)


    utm_coords["time"] = utm_coords.index

    def normalize_time(t):
        return (t - min_t).total_seconds()
    utm_coords["time"] = utm_coords["time"].apply(normalize_time)
    utm_coords.set_index("time", inplace=True)


    if utm_coords.index[0] != 0:
        first_item = utm_coords.iloc[0]

        if debug:
            # set start to left
            y_step = diff_y/num_nodes
            df = pandas.DataFrame({"id":[first_item["id"]],"x":[0],"y":[counter*y_step]}, index=[0], columns=["id","x","y"])
        else:
            # add node at time 0 so the one will recognize it
            df = pandas.DataFrame({"id":[first_item["id"]],"x":[first_item["x"]],"y":[first_item["y"]]}, index=[0], columns=["id","x","y"])

        utm_coords = pandas.concat([df,utm_coords])

    if debug:
        # set end of log to right site
        y_step = diff_y/num_nodes
        utm_coords = utm_coords.set_value(utm_coords.index[-1],"x",diff_x)
        utm_coords = utm_coords.set_value(utm_coords.index[-1],"y",counter*y_step)

        # set gaps to bottom
        x_step = diff_x/num_nodes
        utm_coords.fillna(value={"x":counter*x_step,"y":diff_y},axis="index",inplace=True)

    utm_coords["time"] = utm_coords.index

    return utm_coords.to_dict("records")


# NOTE only supports coords in same UTM zone
def createOneMovements(input_dir,file_ending,worker,output_file,table_name,lat_column="latitude",lon_column="longitude",time_column="timestamp",sim_start=None,rename=False,debug=False):
    output_file = os.path.abspath(output_file)

    if debug:
        logging.info("Debug Mode Activated!")

    if rename:
        # set rename to timestamp to avoid name conflicts during renaming
        rename = int(datetime.now().timestamp())

    logging.info("Reading and processing sqlite files...")

    args = (table_name, lat_column, lon_column, time_column, rename, debug)
    results = forAll(input_dir, file_ending, worker, createOneMovementsWorker, args)

    # remove timestamp in new filename
    if rename:
        for f in os.listdir(input_dir):
            absPath = os.path.abspath(os.path.join(input_dir,f))
            if os.path.isfile(absPath) and absPath.endswith("_%i%s" % (rename, file_ending)):
                os.rename(absPath, absPath.replace("_%i%s" % (rename, file_ending), file_ending))


    min_t = min([r[0] for r in results if r[0]])
    max_t = max([r[1] for r in results if r[1]])
    zone = [r[2] for r in results if r[2]][0]  # will always be the same
    min_x = min([r[3] for r in results if r[3]])
    max_x = max([r[4] for r in results if r[4]])
    min_y = min([r[5] for r in results if r[5]])
    max_y = max([r[6] for r in results if r[6]])

    coords_list = [r[7] for r in results if r[7] is not None]

    diff_x = int(max_x - min_x) + 1
    diff_y = int(max_y - min_y) + 1

    if sim_start is None:
        scale_time = min_t
    else:
        scale_time = sim_start

    scaled_coords = parallelized_apply(coords_list, worker, createOneMovementsScaler, args=(len(coords_list),scale_time,min_x,max_y,diff_x,diff_y,debug))

    logging.info("Sorting and writing movement file...")

    # sort the ids naturally, needed to match the The One GUI ids if -r option is used
    scaled_coords_start = natsorted([r for r in scaled_coords if r["time"] == 0], key=itemgetter("id"))

    scaled_coords_body = sorted([r for r in scaled_coords if r["time"] != 0], key=itemgetter("time"))

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    f = open(output_file,"w")

    # write header: minTime maxTime minX maxX minY maxY minZ maxZ
    content = "%s %s %s %s %s %s 0 0\n" % (0, (max_t - scale_time).total_seconds(), 0, max_x - min_x, 0, max_y - min_y)

    # multithreaded conversion to dict is faster than iterrows() & itertuples()
    for row in scaled_coords_start:
        # time id xPos yPos
        content += "%.3f %s %s %s\n" % (row["time"], row["id"], row["x"], row["y"])

    f.write(content)
    content = ""

    line_counter = 0
    for row in scaled_coords_body:
        # time id xPos yPos
        content += "%.3f %s %s %s\n" % (row["time"], row["id"], row["x"], row["y"])

        if line_counter >= 10000:
            f.write(content)
            content = ""
        line_counter += 1

    f.write(content)

    f.close()

    logging.info("Wrote %s lines with %s different nodes" % (len(scaled_coords),len(coords_list)))
    top_left = utm.to_latlon(min_x, max_y, zone["number"], zone["letter"])
    logging.info("Top-left: Lat/Lon - %.6f, %.6f" % (top_left[0], top_left[1]))
    bottom_right = utm.to_latlon(max_x, min_y, zone["number"], zone["letter"])
    logging.info("Bottom-right: Lat/Lon - %.6f, %.6f" % (bottom_right[0], bottom_right[1]))
    logging.info("Dimensions: %s x %s meters" % (diff_x,diff_y))
    if sim_start is None:
        logging.info("Simulation start time: Used data start time, check out the -s parameter to change this")
    else:
        logging.info("Simulation start time: %s" % sim_start)
    logging.info("Data start time: %s" % min_t)
    logging.info("Data end time: %s" % max_t)
    logging.info("Data duration: %.3f seconds" % (max_t - min_t).total_seconds())
    logging.info("Simulation duration: %.3f seconds" % (max_t - scale_time).total_seconds())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=True, description="This tool converts gps data from multiple sqlite databases into a The ONE compatible movement file. Each database should only contain data of a single node. The database filenames are used as ids.")
    parser.add_argument("-i","--input-dir", type=str, action="store", help="Input directory, containing multiple sqlite database files.", required=True)
    parser.add_argument("-f", "--file-ending", type=str, default=".db", action="store", help="File ending of database files. Default: '.db'")
    parser.add_argument("-o","--output", type=str, default=os.path.join(START_DIR,"movements"), action="store", help="Path of the one movements output file. Default: './movements'")
    parser.add_argument("-w","--worker", type=int, default=4, action="store", help="Number of worker processes, use your core number. Default: 4")
    parser.add_argument("-tb","--table", type=str, action="store", help="Name of the table containing the gps data.", required=True)
    parser.add_argument("-lat","--latitude", type=str, default="latitude", action="store", help="Name of latitude column. Default: 'latitude'")
    parser.add_argument("-lon","--longitude", type=str, default="longitude", action="store", help="Name of longitude column. Default: 'longitude'")
    parser.add_argument("-s","--start", type=str, default=None, action="store", help="Start time of the simulation in 'YYYY-mm-dd HH:MM:SS' format, used to calculate the relative simulation time.")
    parser.add_argument("-t","--time", type=str, default="timestamp", action="store", help="Name of column containing a ISO 8601 timestamp. Default: 'timestamp'")
    parser.add_argument("-r","--rename", default=False, action="store_true", help="Renames your database files to match the continious numeric ids that The One uses in its GUI. Default: Flase")
    parser.add_argument("-d","--debug", default=False, action="store_true", help="Debug flag. Places not started nodes at the left site, nodes with data gaps at the bottom and finished nodes at the right side.")

    ARGS = parser.parse_args()

    if ARGS.start is not None:
        try:
            ARGS.start = datetime.strptime(ARGS.start, "%Y-%m-%d %H:%M:%S")
        except ValueError as e:
            raise ValueError("Could not parse start time. Does it match 'YYYY-mm-dd HH:MM:SS'?")

    createOneMovements(ARGS.input_dir, ARGS.file_ending, ARGS.worker, ARGS.output, ARGS.table, ARGS.latitude, ARGS.longitude, ARGS.time, ARGS.start, ARGS.rename, ARGS.debug)

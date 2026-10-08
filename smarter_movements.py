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

import sys
import os
import logging
from datetime import datetime
from operator import itemgetter
import multiprocessing
import subprocess
import warnings

import numpy
numpy.set_printoptions(threshold=numpy.inf)
import pandas
from pykalman import KalmanFilter
from natsort import natsorted
from geopy.distance import geodesic as geopy_distance
import rdp

import config
import util

import plotUsingMatlib as statistics_functions
# ignore plotly's "Looks like you don't have 'read-write' permission to your 'home' directory" warning which happens during import in multiprocessing workers
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from plotly.offline import plot
    import plotly.graph_objs as go


def smoothMovementsWorker(file, num_iterations, max_gap=None, fromTo=None):
    start_time = datetime.now()

    conn, cur = util.open_sqlite(file)

    gps_data = util.sqlite_to_df(conn, "gps_data", select="timestamp, latitude, longitude", index="timestamp")
    if gps_data is None:
        return

    if fromTo is not None:
        first_item = gps_data.iloc[0]  # use first item as start, because otherwise kalman filter dont know what to do

        df_start = pandas.DataFrame([{"latitude":first_item["latitude"],"longitude":first_item["longitude"]}], index=[fromTo[0]], columns=["latitude","longitude"])
        df_start.index = pandas.to_datetime(df_start.index, format="%Y-%m-%d %H:%M:%S")

        last_item = gps_data.iloc[-1]  # to prevent running in one direction at end

        df_end = pandas.DataFrame([{"latitude":last_item["latitude"],"longitude":last_item["longitude"]}], index=[fromTo[1]], columns=["latitude","longitude"])
        df_end.index = pandas.to_datetime(df_end.index, format="%Y-%m-%d %H:%M:%S")

        gps_data = pandas.concat([df_start,gps_data,df_end])


    gps_data.index.names = ["timestamp"]  # make sure index ist called timestamp

    gps_data = gps_data.resample(rule="1s").mean()

    if max_gap is None:
        data_parts = [gps_data]
    else:
        splits = []
        last_t = None
        i = 0

        for row in gps_data.itertuples():
            if pandas.notnull(row.latitude):
                if last_t is not None and row.Index.timestamp() - last_t > max_gap:
                    splits.append(i)
                last_t = row.Index.timestamp()
            i += 1

        data_parts = numpy.split(gps_data, splits)

    # delete old data
    conn.execute("DROP TABLE IF EXISTS smoothed_gps_data")
    conn.commit()

    if len(data_parts) > 1:
        logging.info("Splitted gps data for %s into %s chunks, because of gaps > %.1f min" % (os.path.basename(file), len(data_parts), max_gap/60))

    for df in data_parts:
        df = df.loc[:df.last_valid_index()]  # cut of NaNs at end, to prevent kalman filter filling gaps in splits

        if len(df) < 2:
            logging.warning("Skipped chunk from %s with < 2 entries" % os.path.basename(file))
            continue

        measurements = numpy.ma.masked_invalid(df[["latitude", "longitude"]].values)

        # x+1  = 1x+0y+1vx+0vy
        # y+1  = 0x+1y+0vx+1vy
        # vx+1 = 0x+0y+1vx+0vy
        # vy+1 = 0x+0y+0vx+1vy
        F = numpy.array([[1, 0, 1, 1],
                         [0, 1, 0, 1],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]])

        # x = 1x+0y+0vx+0vy
        # x = 0x+1y+0vx+0vy
        H = numpy.array([[1, 0, 0, 0],
                         [0, 1, 0, 0]])

        R = numpy.diag([1e-4, 1e-4])**2  # 10-30m

        # use first measurement and 0 for velocity
        initial_state_mean = numpy.hstack([measurements[0, :], 2*[0.]])
        initial_state_covariance = numpy.diag([1e-4, 1e-4, 1e-6, 1e-6])**2

        kf = KalmanFilter(transition_matrices=F,
                          observation_matrices=H,
                          observation_covariance=R,
                          initial_state_mean=initial_state_mean,
                          initial_state_covariance=initial_state_covariance,
                          em_vars=["transition_covariance"])

        try:
            kf = kf.em(measurements, n_iter=num_iterations)
        except Exception as e:
            print(df)
            print(file)
            print(measurements)
            raise e

        state_means, state_vars = kf.smooth(measurements)

        pandas.options.mode.chained_assignment = None  # disable SettingWithCopyWarning, default='warn'
        df.ix[:, ["latitude", "longitude"]] = state_means[:,:2]
        pandas.options.mode.chained_assignment = "warn"

        df.to_sql("smoothed_gps_data", conn, if_exists="append", index=True)

    conn.close()

    logging.debug("smoothMovementsWorker finished for %s in %s" % (os.path.basename(file), (datetime.now()-start_time)))

    return


def smoothMovements(input_dir, num_iterations, max_gap=None, fromTo=None):

    util.forAllMP(input_dir,".db",smoothMovementsWorker, (num_iterations, max_gap, fromTo))


def calcSpeedsWorker(file, tb_name):
    conn, cur = util.open_sqlite(file)

    gps_data = util.sqlite_to_df(conn, tb_name, parse_s=["timestamp"])
    if gps_data is None:
        return

    gps_data.sort_values("timestamp",inplace=True)

    results = []
    prev = None
    for row in gps_data[["timestamp", "latitude", "longitude"]].values:
        if prev is None:
            results.append([numpy.nan, numpy.nan])
        else:
            distance = geopy_distance((prev[1], prev[2]), (row[1], row[2])).meters
            #logging.info("############# DISTANCE: %s - geopy_distance((prev[%s], prev[%s]), (row[%s], row[%s])).meters", distance, prev[1],prev[2],row[1], row[2])
            try:
                #logging.info("Calculating speed: %s / %s - %s" % (distance, str(row[0]), str(prev[0])))
                if not pandas.isnull(row[0]):
                    speed = distance / (row[0].timestamp() - prev[0].timestamp())
                else:
                    speed = numpy.nan
            except ZeroDivisionError:
                speed = numpy.nan
            results.append([distance, speed])
        prev = row

    gps_data["distance"] = [r[0] for r in results]
    gps_data["speed"] = [r[1] for r in results]

    # TODO TEMP FIX?
    gps_data.to_sql(tb_name, conn, if_exists="replace", index=False)
    conn.close()


def calcSpeeds(input_dir, gps_tables):

    for table in gps_tables:
        util.forAllMP(input_dir,".db", calcSpeedsWorker, (table,))


def plotGPSTracksWorker(file, filter_fromTo, simplify, gps_tables, plots_dir=None):
    conn, cur = util.open_sqlite(file)

    data = []
    traces = []

    if plots_dir:
        os.makedirs(plots_dir, exist_ok=True)

    if filter_fromTo:
        where = "DATETIME(timestamp) BETWEEN DATETIME('%s') AND DATETIME('%s')" % (filter_fromTo[0], filter_fromTo[1])
    else:
        where = None

    for table in gps_tables:

        gps_data = util.sqlite_to_df(conn, table, select="timestamp, latitude, longitude", index="timestamp", where=where)
        if gps_data is None:
            return

        gps_data = gps_data.resample("1s").mean()  # resample to generate nans

        if simplify:
            # simplify for javascript
            gps_data_mask = rdp.rdp(gps_data[["latitude", "longitude"]].values, 0.0001, algo="iter", return_mask=True)

            simplified_gps_data = gps_data[gps_data_mask | numpy.isnan(gps_data["latitude"])]  # keep nans

            logging.info("%s - %s gps points simplified to %s" % (os.path.basename(file), len(gps_data), numpy.count_nonzero(gps_data_mask)))

            gps_data = simplified_gps_data

        gps_data = gps_data[gps_data.ffill(limit=1).notnull()["latitude"]]  # just on nan per gap

        # have to use data for return because Scattermapbox throws error with multiprocessing
        data.append({"name":"%s_%s" % (os.path.basename(file), table),"lat":gps_data["latitude"].values, "lon":gps_data["longitude"].values})

        if plots_dir:
            trace = go.Scattermapbox(
                    name="%s_%s" % (os.path.basename(file), table),
                    mode="lines+markers",
                    connectgaps=False,
                    lat=gps_data["latitude"].values,
                    lon=gps_data["longitude"].values,
                    opacity=0.7,
                    line=go.Line(width=0.5, color="#888"),
                    marker=go.Marker(color="rgb(255, 0, 0)"),
                    hoverinfo="lat+lon+name"
                    )

            traces.append(trace)


    if plots_dir:
        layout = go.Layout(title="GPS Tracks of %s" % os.path.basename(file), 
                           #font=dict(size=16), 
                           showlegend=True, mapbox=dict(accesstoken=config.MAPBOX_TOKEN, zoom=8, center=dict(lat=51.81932, lon=8.77124)))
        fig = go.Figure(data=traces, layout=layout)
        plot(fig, filename=os.path.join(plots_dir, "gps_tracks_%s.html" % os.path.splitext(os.path.basename(file))[0]), show_link=False, auto_open=False)

    return data


def plotGPSTracks(input_dir, plots_dir, filter_fromTo=None, simplify=True, file_ending=".db", only=None, plot_all=False, gps_tables=None, auto_open=True):

    os.makedirs(plots_dir, exist_ok=True)

    traces = []

    if only:
        data = plotGPSTracksWorker(os.path.join(input_dir, only), filter_fromTo, simplify, gps_tables)

        title = "GPS Tracks of %s" % only
        filename = "gps_tracks_%s.html" % only
    else:
        if plot_all:
            args = (simplify, gps_tables, os.path.join(plots_dir, "gps_tracks"))
        else:
            args = (filter_fromTo, simplify, gps_tables,)

        results = util.forAllMP(input_dir, file_ending, plotGPSTracksWorker, args)

        data = []
        for result in results:
            for d in result:
                data.append(d)

        title = "GPS Tracks of %s files" % len(results)
        filename = "gps_tracks.html"


    data = natsorted(data, key=itemgetter("name"))

    for d in data:
        traces.append(go.Scattermapbox(
                name=d["name"],
                mode="lines+markers",
                connectgaps=False,
                lat=d["lat"],
                lon=d["lon"],
                opacity=0.7,
                line=go.Line(width=0.5, color="#888"),
                marker=go.Marker(color="rgb(255, 0, 0)"),
                hoverinfo="lat+lon+name"
                ))

    layout = go.Layout(title=title, 
                       #font=dict(size=16), 
                       showlegend=True, mapbox=dict(accesstoken=config.MAPBOX_TOKEN, zoom=8, center=dict(lat=51.81932, lon=8.77124)))

    layout["updatemenus"] = [dict(buttons=[dict(label="show all", method="restyle", args=["visible", True]), dict(label="hide all", method="restyle", args=["visible", "legendonly"])])]

    fig = go.Figure(data=traces, layout=layout)

    plot(fig, filename=os.path.join(plots_dir, filename), show_link=False, auto_open=auto_open)


def plotSpeedsWorker(file, tb_name, resample, filter_fromTo):

    if filter_fromTo:
        where = "DATETIME(timestamp) BETWEEN DATETIME('%s') AND DATETIME('%s')" % (filter_fromTo[0], filter_fromTo[1])
    else:
        where = None

    df = util.sqlite_to_df(file, tb_name, select="timestamp, speed", where=where, index="timestamp")
    if df is None:
        return

    df.sort_index(inplace=True)

    df = df.resample(rule=resample).mean()

    df["speed"] = df["speed"] * 3.6  # km/h

    df = df[df.ffill(limit=1).notnull()["speed"]]  # just on nan per gap

    return {"name":"%s" % os.path.basename(file),"df":df}


def plotSpeeds(input_dir, plots_dir, gps_tables, filter_fromTo=None, auto_open=True):

    os.makedirs(plots_dir, exist_ok=True)

    RESAMPLE = "10s"

    table_results = {}

    for table in gps_tables:

        data = util.forAllMP(input_dir, ".db", plotSpeedsWorker, (table, RESAMPLE, filter_fromTo))

        data = natsorted(data, key=itemgetter("name"))

        table_results[table] = data

        # Speed Time Series ###
        traces = []
        for d in data:
            traces.append(go.Scatter(x=d["df"].index, y=d["df"]["speed"], name=d["name"]))

        layout = go.Layout(title="<b>Speed Time Series of %s Nodes (%s slots)</b>" % (len(data), RESAMPLE)#,
                           #yaxis=dict(
                           #    title="km/h", 
                           #    #font=dict(size=19), 
                           #    #tickfont=dict(size=19)
                           #    ),
                           ##font=dict(size=22),
                           #xaxis=dict(
                           #    #font=dict(size=19), 
                           #    #tickfont=dict(size=19), 
                           #    type="date")#,
                           ##legend=dict(font=dict(size=18))
                           )

        layout["updatemenus"] = [dict(buttons=[dict(label="show all", method="restyle", args=["visible", True]), dict(label="hide all", method="restyle", args=["visible", "legendonly"])])]

        fig = go.Figure(data=traces, layout=layout)
        plot(fig, filename=os.path.join(plots_dir,"speed_timeseries_%s.html" % table), show_link=False, auto_open=auto_open)
        ###

    # Speed histo ###
    max_speed = 10

    traces = []
    sub_title = ""

    table2label = {"gps_data":"unmodified gps data"} #, "smoothed_gps_data": "smoothed gps data"}

    for table in gps_tables:
        speed_dfs = [d["df"] for d in table_results[table]]
        speeds = [v for df in speed_dfs for v in df["speed"].values]

        scaled_speeds = [s if s < max_speed else max_speed + 0.1 for s in speeds]

        sub_title += "<br>avg. %s: %.2f km/h" % (table2label[table], numpy.nanmean(speeds))

        trace = go.Histogram(hoverinfo="y+x", name=table2label[table], x=scaled_speeds, opacity=0.75, autobinx=False, xbins=dict(start=0,end=max_speed+1,size=0.1), histnorm="percent")
        traces.append(trace)

    layout = go.Layout(title="<b>Walking Speed Distibution</b>%s" % sub_title#, 
        #font=dict(size=27), 
        #xaxis=dict(
        #    title="Speed in km/h", 
        #    #font=dict(size=24), 
        #    #tickfont=dict(size=24)
        #), 
        #yaxis=dict(
        #    title="%% of time (%s slots)" % RESAMPLE, 
        #    #font=dict(size=24), 
        #    #tickfont=dict(size=24)
        #)#, 
        #bargap=0.1#, legend=dict(font=dict(size=24))
        )


    fig = go.Figure(data=traces, layout=layout)

    plot(fig, filename=os.path.join(plots_dir,"speed_distribution.html"), show_link=False, auto_open=auto_open)
    ###


def poltWalkingDistanceWorker(file, tb_name, filter_fromTo):

    if filter_fromTo:
        where = "DATETIME(timestamp) BETWEEN DATETIME('%s') AND DATETIME('%s')" % (filter_fromTo[0], filter_fromTo[1])
    else:
        where = None

    df = util.sqlite_to_df(file, tb_name, select="distance", where=where)
    if df is None:
        return

    walked = df["distance"].sum()

    return {"name":"%s" % os.path.basename(file),"walked":walked/1000}


def poltWalkingDistance(input_dir, plots_dir, gps_tables, filter_fromTo=None, auto_open=True):

    os.makedirs(plots_dir, exist_ok=True)

    table2label = {"gps_data":"unmodified gps data"} # , "smoothed_gps_data": "smoothed gps data"}

    traces = []
    sub_title = ""

    histo_sub_title = ""
    histo_traces = []

    for table in gps_tables:

        data = util.forAllMP(input_dir, ".db", poltWalkingDistanceWorker, (table, filter_fromTo))

        data = natsorted(data, key=itemgetter("name"))

        x = [d["name"].split("_")[0] for d in data]
        y = [d["walked"] for d in data]

        # Walking distance Bar Chart ###
        if len(y) == 0: 
            continue
        sub_title += "<br>avg. %s: %.2f km" % (table2label[table], sum(y)/len(y))
        traces.append(go.Bar(x=x, y=y, name=table2label[table], opacity=0.75))

        # Histo
        histo_sub_title += "<br>avg. %s: %.2f km" % (table2label[table], sum(y)/len(y))
        histo_traces.append(go.Histogram(hoverinfo="y+x", name=table2label[table], x=y, opacity=0.75, autobinx=False, xbins=dict(start=0,end=int(max(y))+1,size=1)))

    # Walking distance Bar Chart ###
    layout = go.Layout(title="<b>Walking Distances of %s Nodes</b>%s" % (len(x), sub_title)#, 
        #font=dict(size=24), 
        #xaxis=dict(
        #    title="Node", 
            #font=dict(size=22), 
            #tickfont=dict(size=22)
        #), 
        #yaxis=dict(
        #    title="Kilometer", 
            #font=dict(size=22), 
            #tickfont=dict(size=22)
        #) 
       #bargap=0.2#, 
        #legend=dict(
        #    font=dict(size=22)
        #)
    )

    fig = go.Figure(data=traces, layout=layout)

    plot(fig, filename=os.path.join(plots_dir,"walking_distances.html"), show_link=False, auto_open=auto_open)

    # Histo
    layout = go.Layout(title="<b>Walking Distances of %s Nodes</b>%s" % (len(x), histo_sub_title) 
        #font=dict(size=27), 
        #xaxis=dict(
        #    title="Kilometer", 
            #font=dict(size=24), 
            #tickfont=dict(size=24)
        #), 
        #yaxis=dict(
        #    title="Number of Nodes", 
            #font=dict(size=24), 
            #tickfont=dict(size=24)
        #)#, 
        #bargap=0.1#, 
        #legend=dict(font=dict(size=24))
    )

    fig = go.Figure(data=histo_traces, layout=layout)

    plot(fig, filename=os.path.join(plots_dir,"walking_distances_histo.html"), show_link=False, auto_open=auto_open)


def countNeighborsResampler(file, gps_table, fromTo, resample):
    coords = util.sqlite_to_df(file, gps_table, select="timestamp, latitude, longitude", index="timestamp")
    if coords is None:
        return
    
    #logging.warning("countNeighborsResampler in action file: %s, gps_table %s, fromTo %s, resample %s" %(file, gps_table, fromTo, resample))
      # make sure resamples over all nodes have same start & end
    df_start = pandas.DataFrame([{"latitude":numpy.nan,"longitude":numpy.nan}], index=[fromTo[0]], columns=["latitude","longitude"])
    df_start.index = pandas.to_datetime(df_start.index, format="%Y-%m-%d %H:%M:%S")

    df_end = pandas.DataFrame([{"latitude":numpy.nan,"longitude":numpy.nan}], index=[fromTo[1]], columns=["latitude","longitude"])
    df_end.index = pandas.to_datetime(df_end.index, format="%Y-%m-%d %H:%M:%S")

    coords = pandas.concat([df_start,coords,df_end])

    coords = coords.resample(rule=resample).mean()
    
        #for i in enumerate(coords["latitude"].values):
    #    logging.warning("Coords[latitude].values loop - file %s - entry %f" % (file, i[1]))
    return {"name":os.path.basename(file), "coords":coords[["latitude","longitude"]].values}


def countNeighborsWorker(coords, max_distance):

    neighbors = []

    for i,pos in enumerate(coords):
        n_count = 0

        for j,other in enumerate(coords):
            if i != j:
                if not numpy.isnan(pos[0]) and not numpy.isnan(pos[1]) and not numpy.isnan(other[0]) and not numpy.isnan(other[1]):
                    distance = geopy_distance((pos[0], pos[1]), (other[0], other[1])).meters
                    #logging.warning("Checking distance for %i and %i - result : %i" % (i,j,distance))

                    if distance <= max_distance:
                        #logging.warning("Adding neighbor - amount neighbors:%s" % n_count)
                        n_count += 1
        #logging.warning("Number neighbors: n_count: %i for node %i - pos[0] %s post[1] %s" % (n_count, i, pos[0], pos[1]))
        neighbors.append(n_count)

    return neighbors


def countNeighbors(input_dir, output_dir, gps_tables, fromTo, resample, max_distance):

    os.makedirs(output_dir, exist_ok=True)

    start_time = datetime.now()

    conn, cur = util.open_sqlite(os.path.join(output_dir,"neighbors.db"), create=True)

    cur.execute("DROP TABLE IF EXISTS neighbors_info")
    cur.execute("CREATE TABLE neighbors_info (max_distance REAL, resample TEXT, PRIMARY KEY (max_distance,resample))")
    cur.execute("INSERT INTO neighbors_info (max_distance, resample) VALUES (?,?)", [max_distance, resample])
    conn.commit()

    for table in gps_tables:
        resampled_coords = util.forAllMP(input_dir,".db", countNeighborsResampler, (table, fromTo, resample))

        resampled_coords = natsorted(resampled_coords, key=itemgetter("name"))

        node_names = [r["name"] for r in resampled_coords]
        time_index = pandas.date_range(start=fromTo[0], end=fromTo[1], freq=resample, name="timestamp")

        # reshape rows time & columns nodes
        coords_data = []
        for i in range(0, len(time_index)):
            coords_data.append([r["coords"][i] for r in resampled_coords])

        start_worker = datetime.now()

        logging.warning("Starting countNeighborsWorker ...")

        pool = multiprocessing.Pool(processes=config.POOL_SIZE)

        results = []
        processes = []

        for row in coords_data:
            #logging.warning("coords_data entry : %s" % row)
            p = pool.apply_async(func=util.worker_wrapper, args=(countNeighborsWorker, (row, max_distance)))
            processes.append(p)

        for p in processes:
            try:
                result = p.get()
            except KeyboardInterrupt:
                logging.info("Got ^C terminating...")
                pool.terminate()
                sys.exit(-1)
            if result is not None:
                results.append(result)

        pool.close()

        logging.debug("All countNeighborsWorker finished in %s" % (datetime.now()-start_worker))

        df = pandas.DataFrame(data=results, index=time_index, columns=node_names)

        df.to_sql("neighbors_%s" % table, conn, if_exists="replace", index=True)

    conn.close()

    logging.debug("countNeighbors finished in %s" % (datetime.now()-start_time))


def plotNeighbors(data_path, plots_dir, gps_tables, filter_fromTo=None, auto_open=True):

    os.makedirs(plots_dir, exist_ok=True)

    avg_traces = []
    node_traces = []
    info = util.sqlite_to_df(os.path.join(data_path,"neighbors.db"), "neighbors_info")

    if filter_fromTo:
        where = "DATETIME(timestamp) BETWEEN DATETIME('%s') AND DATETIME('%s')" % (filter_fromTo[0], filter_fromTo[1])
    else:
        where = None

    for table in gps_tables:
        df = util.sqlite_to_df(os.path.join(data_path,"neighbors.db"), "neighbors_%s" % table, index="timestamp", where=where)

        df["avg"] = df.mean(axis="columns")
        avg_traces.append(go.Scatter(x=df.index, y=df["avg"], name="avg %s" % table))

        avg_nodes = df.mean(axis="index")
        node_traces.append(go.Bar(x=df.columns, y=avg_nodes, name=table))

        # Neighbors Time Series ###
        traces = []

        traces.append(go.Scattergl(x=df.index, y=df["avg"], name="All Average"))

        for column in df.columns:
            if column != "avg":
                traces.append(go.Scattergl(x=df.index, y=df[column], name=column))

        layout = go.Layout(title="<b>Number of Neighbors (%s Nodes, %s slots, %s m)</b>" % (len(df.columns)-1, info["resample"][0], info["max_distance"][0])
                           #yaxis=dict(title="# Neighbors"),
                           #font=dict(size=22),
                           #xaxis=dict(type="date"),
                           #legend=dict(font=dict(size=18))
                           )

        layout["updatemenus"] = [dict(buttons=[dict(label="show all", method="restyle", args=["visible", True]), dict(label="hide all", method="restyle", args=["visible", "legendonly"])])]

        fig = go.Figure(data=traces, layout=layout)
        ###plot(fig, filename=os.path.join(plots_dir,"neighbors_timeseries_%s.html" % table), show_link=False, auto_open=auto_open)
        ###

    # Bar Chart Node avg ###
    layout = go.Layout(title="<b>Avg Neighbors of %s Nodes</b>" % (len(df.columns)-1)
                       #font=dict(size=24), 
                       #xaxis=dict(title="Node"#, 
                                  #font=dict(size=22), tickfont=dict(size=22)
                       #           ), 
                       # yaxis=dict(title="# Neighbors")#, 
                                #font=dict(size=22),
                                #tickfont=dict(size=22)), 
                                #bargap=0.1) #, legend=dict(font=dict(size=22))
                        )

    fig = go.Figure(data=node_traces, layout=layout)

    ###plot(fig, filename=os.path.join(plots_dir,"neighbors_node_avg.html"), show_link=False, auto_open=auto_open)
    ###

    # Avg Time Series ###
    layout = go.Layout(title="<b>Avg Number of Neighbors (%s slots, %s m)</b>" % (info["resample"][0], info["max_distance"][0])
                       #yaxis=dict(title="# Neighbors", 
                                  #font=dict(size=19), 
                                  #tickfont=dict(size=19)),
                       #font=dict(size=22),
                       #xaxis=dict(rangeselector=dict(buttons=list([
                       #                                             dict(count=30,
                       #                                                  label='30min',
                       #                                                  step='minute',
                       #                                                  stepmode='backward'),
                       #                                             dict(count=1,
                       #                                                  label='1h',
                       #                                                  step='hour',
                       #                                                  stepmode='backward'),
                       #                                             dict(count=4,
                       #                                                  label='4h',
                       #                                                  step='hour',
                       #                                                  stepmode='backward'),
                       #                                             dict(step='all')
                       #                                         ])
                       #                              ),
                       #           rangeslider=dict(),
                       #           type="date")#,
                       #legend=dict(font=dict(size=18))
                       )#)

    fig = go.Figure(data=avg_traces, layout=layout)
    ##plot(fig, filename=os.path.join(plots_dir,"neighbors_time_avg.html"), show_link=False, auto_open=auto_open)
    ###
    return avg_traces



def writeFile(input_dir, output_dir, gps_tables, fromTo=False, debug=False):

    for table in gps_tables:
        args = [sys.executable,"sqlite2OneMovements.py","-i",input_dir,"-o",os.path.join(output_dir,"movements_%s" % table),"-tb",table,"-w","%s" % config.POOL_SIZE]

        if debug:
            args += ["-d"]
        if fromTo:
            args += ["-s", fromTo[0]]

        subprocess.call(args)


if __name__ == "__main__":

    gps_tables = ["gps_data"] # , "smoothed_gps_data"]

    # use a kalman filter to smooth the gps data and fill gaps
    # smoothMovements(config.DATA_DIR, config.KALMAN_ITERATIONS, config.MAX_SMOOTH_GAP)

    # write movements from db into The One compatible file
    #writeFile(config.DATA_DIR, config.DATA_PATH, gps_tables, config.FROM_TO, debug=False)

    # calculate and plot speeds (& walking distances)
    calcSpeeds(config.DATA_DIR, gps_tables)
    plotSpeeds(config.DATA_DIR, config.PLOTS_DIR, gps_tables, filter_fromTo=config.FROM_TO, auto_open=config.AUTO_OPEN_PLOTS)
    poltWalkingDistance(config.DATA_DIR, config.PLOTS_DIR, gps_tables, filter_fromTo=config.FROM_TO, auto_open=config.AUTO_OPEN_PLOTS)

    # plotGPSTracks(config.DATA_DIR, os.path.join(config.PLOTS_DIR,"gps_tracks"), only="25_192.168.0.26.db", gps_tables=gps_tables, simplify=False)
    plotGPSTracks(config.DATA_DIR, config.PLOTS_DIR, filter_fromTo=config.FROM_TO, file_ending=".db", gps_tables=gps_tables, simplify=config.GPS_SIMPLIFY, plot_all=config.PLOT_ALL_GPS_TRACKS, auto_open=config.AUTO_OPEN_PLOTS)

    # create neighbors.db by counting all nodes wihin MAX_NEIGHBOR_DISTANCE around node (this does not checks active connections)
    #countNeighbors(config.DATA_DIR, config.DATA_PATH, gps_tables, config.FROM_TO, resample="10s", max_distance=config.MAX_NEIGHBOR_DISTANCE)
    countNeighbors(config.DATA_DIR, config.DATA_PATH, gps_tables, config.FROM_TO, resample="120s", max_distance=25)
    avg25m = plotNeighbors(config.DATA_PATH, config.PLOTS_DIR, gps_tables, filter_fromTo=config.FROM_TO, auto_open=config.AUTO_OPEN_PLOTS)

    countNeighbors(config.DATA_DIR, config.DATA_PATH, gps_tables, config.FROM_TO, resample="120s", max_distance=44)
    avg44m = plotNeighbors(config.DATA_PATH, config.PLOTS_DIR, gps_tables, filter_fromTo=config.FROM_TO, auto_open=config.AUTO_OPEN_PLOTS)

    countNeighbors(config.DATA_DIR, config.DATA_PATH, gps_tables, config.FROM_TO, resample="120s", max_distance=110)
    avg110m = plotNeighbors(config.DATA_PATH, config.PLOTS_DIR, gps_tables, filter_fromTo=config.FROM_TO, auto_open=config.AUTO_OPEN_PLOTS)

    xdata = [avg25m[0].x, avg44m[0].x, avg110m[0].x]
    ydata = [avg25m[0].y, avg44m[0].y, avg110m[0].y]


    print("#### Plot DATA - 25m ###")
    print("### X Data ###")
    for i in avg25m[0].x:
        print("%s" % i)
    print("### Y Data ###")
    for i in avg25m[0].x:
        print("%f" % i)
    
    print("#### Plot DATA - 44m ###")
    print("### X Data ###")
    for i in avg44m[0].x:
        print("%s" % i)
    print("### Y Data ###")
    for i in avg44m[0].x:
        print("%f" % i)
    
    print("#### Plot DATA - 110m ###")
    print("### X Data ###")
    for i in avg110m[0].x:
        print("%s" % i)
    print("### Y Data ###")
    for i in avg110m[0].x:
        print("%f" % i)
    #print("x1: %s" % clusterX)
    #print("y1: %s" % clusterY)
    print ("#### END Plot DATA ###")

    #statistics_functions.timeUsingMatplot(xdata, ydata, "dataX", "dataY")
    statistics_functions.ecdfNeighborDistance(ydata)
    # plot stats about neighbors
    # plotNeighbors(config.DATA_PATH, config.PLOTS_DIR, gps_tables, filter_fromTo=config.FROM_TO, auto_open=config.AUTO_OPEN_PLOTS)
    
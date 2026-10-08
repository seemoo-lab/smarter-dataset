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
import logging
import sqlite3
import multiprocessing
import re
import math
from operator import itemgetter
from datetime import datetime
from datetime import timedelta
import warnings
from collections import Counter

import pandas
import numpy
numpy.set_printoptions(threshold=numpy.inf)
from geopy import distance
import networkx

import config
import util
import plotUsingMatlib as statistics_functions

import scipy.stats as stats
import matplotlib.pyplot as plt

# ignore plotly's "Looks like you don't have 'read-write' permission to your 'home' directory" warning which happens during import in multiprocessing workers
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from plotly.offline import plot
    import plotly.graph_objs as go

###### start creation of the connection.db #####
def createOneConnectionsWorker(file, ids, max_distance, gps_lookup_offset, gps_table):
    start_time = datetime.now()

    _id = int(os.path.splitext(os.path.basename(file))[0].split("_")[0])

    ip = util.getIp(ids, id=_id)
    dtn_ids = util.getDtnIds(ids, id=_id)

    if ip is None or dtn_ids is None:
        logging.error("Could not get ip or dtn_ids for %s" % _id)
        return

    conn, cur = util.open_sqlite(file)

    connection_messages = util.query_sqlite(cur, "SELECT message, class, timestamp FROM log_data WHERE class=?", ["NodeEvent"], if_no_table=[])

    ups = 0
    downs = 0
    connections = []
    dbs = {}

    for row in connection_messages:

        match = re.match("Node dtn://(android-[a-z0-9]+)\.dtn (available|unavailable)", row["message"])
        if match:

            if match.group(2) == "available":
                ups += 1
            else:
                downs += 1

            to_id = util.getId(ids, dtn_id=match.group(1))
            to_ip = util.getIp(ids, dtn_id=match.group(1))

            if to_id is not None and to_ip is not None:
                # save all db connections
                db = dbs.get(to_id, None)
                if not db:
                    node_conn, node_cur = util.open_sqlite(os.path.join(os.path.dirname(file),"%s_%s.db" % (to_id, to_ip)))
                    db = {"conn": node_conn, "cur": node_cur}
                    dbs[to_id] = db


                state_map = {"available":"up","unavailable":"down"}

                connection = {"timestamp":row["timestamp"], "fromId":_id, "toId":to_id, "state":state_map[match.group(2)], "distance": None}

                delta = timedelta(seconds=gps_lookup_offset)
                start = row["timestamp"] - delta
                stop = row["timestamp"] + delta

                query = "SELECT latitude AS lat, longitude AS lon FROM %s WHERE timestamp BETWEEN ? AND ?" % gps_table

                from_gps_data = util.query_sqlite(cur, query, [start, stop], if_no_table=[])

                to_gps_data = util.query_sqlite(db["cur"], query, [start, stop], if_no_table=[])


                if len(from_gps_data) > 0 and len(to_gps_data) > 0:
                    from_lat = numpy.mean([i["lat"] for i in from_gps_data])
                    from_lon = numpy.mean([i["lon"] for i in from_gps_data])

                    to_lat = numpy.mean([i["lat"] for i in to_gps_data])
                    to_lon = numpy.mean([i["lon"] for i in to_gps_data])

                    connection["distance"] = distance.geodesic((from_lat, from_lon), (to_lat, to_lon)).meters

                    if connection["distance"] > max_distance:
                        logging.debug("Distance %.2f for %s -> %s %s at %s - %s" % (connection["distance"],connection["fromId"],connection["toId"],connection["state"],start,stop))


                connections.append(connection)

            else:
                logging.error("Could not find to_id or to_ip for %s, came from db with id %s" % (match.group(1), _id))


    for index, db in dbs.items():
        db["conn"].close()
    conn.close()

    logging.debug("Filtered %s ConnectionEvent lines for id %s. ups: %s, downs: %s in %.2f seconds" % (len(connection_messages), _id, ups, downs, (datetime.now()-start_time).total_seconds()))


    if len(connections) <= 0:
        return
    else:
        return connections

    # CONN up Event?
    # TCPConnection, Initiate TCP connection to 192.168.0.90:4556
    # TCPConvergenceLayer, tcp connection added (tcp://192.168.0.55:57168)
    # ConnectionManager, New node available: Node: dtn://android-f65b27c7.dtn [ 0#discovered#TCP#ip=192.168.0.102;port=4556;#expire=557655578;  ]
    # ConnectionManager, New node available: Node: dtn://android-7100064b.dtn [ 10#connected#TCP#ip=192.168.0.61;port=60464;#expire=0;  ]
    # NodeEvent, Node dtn://android-20307adf.dtn available
    # ConnectionEvent, connection setup dtn://android-e7d88fdf.dtn
    # ConnectionEvent, connection up dtn://android-e7d88fdf.dtn

    # LinkManager info about interfaces, action = up or down ?

    # CONN down Event?
    # TCPConnection, connection to dtn://android-e7d88fdf.dtn failed
    # TCPConvergenceLayer, tcp connection removed (dtn://android-1ab40a4d.dtn)
    # NodeEvent, Node dtn://android-b8cc1899.dtn unavailable
    # ConnectionEvent, connection down dtn://android-69199fa8.dtn

def createOneConnections(data_path, data_sub_dir, max_distance, conn_lookup_offset, gps_lookup_offset, gps_table):
    # just read in ids.db once
    ids = util.readIds(os.path.join(data_path,"ids.db"))

    results = util.forAllMP(os.path.join(data_path, data_sub_dir), ".db", createOneConnectionsWorker, (ids, max_distance, gps_lookup_offset, gps_table))

    results_list = [item for r in results for item in r]  # flatten results

    raw_connections = sorted(results_list, key=itemgetter("timestamp"))


    conn, cur = util.open_sqlite(os.path.join(data_path,"connections.db"), create=True, max_speed=True)
    cur.execute("DROP TABLE IF EXISTS connection_events")
    cur.execute("CREATE TABLE connection_events (timestamp TIMESTAMP NOT NULL, conn_id TEXT NOT NULL, fromId INTEGER NOT NULL, toId INTEGER NOT NULL, state TEXT NOT NULL, distance REAL, flag TEXT NOT NULL, PRIMARY KEY (timestamp, conn_id, state, distance, flag))")
    cur.execute("DROP TABLE IF EXISTS connections")
    cur.execute("CREATE TABLE connections (start_t TIMESTAMP, end_t TIMESTAMP, duration REAL, conn_id TEXT NOT NULL, id1 INTEGER, id2 INTEGER, up_distance REAL, down_distance REAL, flag TEXT, PRIMARY KEY (start_t, end_t, conn_id))")

    conn.commit()
    cur.execute("BEGIN")

    # filter double opens
    for c in raw_connections:

        flag = None

        sorted_ids = sorted([c["fromId"], c["toId"]])
        conn_id = "%s-%s" % (sorted_ids[0], sorted_ids[1])  # id where the direction is irrelevant

        if c["state"] in ["up","down"]:

            if c["distance"] and c["distance"] > max_distance and c["state"] == "up":
                flag = "max_distance_up"
            else:
                delta = timedelta(seconds=conn_lookup_offset)
                values = [c["timestamp"]-delta, c["timestamp"]+delta, c["state"], conn_id]
                cur.execute("SELECT * FROM connection_events WHERE (timestamp BETWEEN ? AND ?) AND state=? AND conn_id=? ORDER BY timestamp", values)
                result = cur.fetchone()

                if result:
                    if c["fromId"] == result["toId"]:
                        flag = "direction"
                    else:
                        flag = "repeat"
                else:

                    if c["state"] == "up":
                        cur.execute("SELECT * FROM connections WHERE conn_id=? AND start_t<=? AND end_t IS NULL", [conn_id, c["timestamp"]])
                        result = cur.fetchone()

                        if result:
                            flag = "not_closed"  # maybe implicit close here
                        else:
                            cur.execute("INSERT INTO connections (start_t, conn_id, id1, id2, up_distance, flag) VALUES (?,?,?,?,?,?)", [c["timestamp"], conn_id, c["fromId"], c["toId"], c["distance"], "create"])
                            flag = "ok"

                    elif c["state"] == "down":
                        cur.execute("SELECT rowid, * FROM connections WHERE conn_id=? AND start_t<=? AND end_t IS NULL", [conn_id, c["timestamp"]])
                        result = cur.fetchone()

                        if result:
                            duration = (c["timestamp"]-result["start_t"]).total_seconds()
                            cur.execute("UPDATE connections SET end_t=?, duration=?, down_distance=? WHERE rowid=?", [c["timestamp"], duration, c["distance"], result["rowid"]])

                            if c["distance"] and c["distance"] > max_distance:
                                flag = "ok_max_distance_down"  # check exact distances of connections later
                            else:
                                flag = "ok"
                        else:
                            flag = "not_open"  # implicit open?

            try:
                values = [c["timestamp"], conn_id, c["fromId"], c["toId"], c["state"], c["distance"], flag]
                cur.execute("INSERT INTO connection_events (timestamp, conn_id, fromId, toId, state, distance, flag) VALUES (?,?,?,?,?,?,?)", values)  # will be ignored sometimes
            except sqlite3.IntegrityError as e:
                if "UNIQUE constraint failed" in str(e):
                    logging.error("UNIQUE constraint failed: %s, %s -> %s, %s, %s, %s" % (c["timestamp"], c["fromId"], c["toId"], c["state"], c["distance"], flag))
                else:
                    raise e

        else:
            pass  # ignore setups


    conn.commit()

    # save raw connections to db
    raw_df = pandas.DataFrame(raw_connections, columns=raw_connections[0].keys())
    raw_df.to_sql("raw_connections", conn, if_exists="replace", index=False)

    raw_up = raw_df[raw_df["state"] == "up"]
    raw_down = raw_df[raw_df["state"] == "down"]

    logging.info("Raw: Mean up distance: %.2f based on #%s (all = %s)" % (raw_up["distance"].mean(), raw_up["distance"].count(), len(raw_up)))
    logging.info("Raw: Mean down distance: %.2f based on #%s (all = %s)" % (raw_down["distance"].mean(), raw_down["distance"].count(), len(raw_down)))

    df = util.sqlite_to_df(conn, "connection_events", where="flag LIKE 'ok%'", order="timestamp", index="timestamp")
    conn.close()

    up = df[df["state"] == "up"]
    down = df[df["state"] == "down"]

    logging.info("Filtered (ok%%): Mean up distance: %.2f based on #%s (all = %s)" % (up["distance"].mean(), up["distance"].count(), len(up)))
    logging.info("Filtered (ok%%): Mean down distance: %.2f based on #%s (all = %s)" % (down["distance"].mean(), down["distance"].count(), len(down)))

def checkGPSDistanceWorker(data_path, data_sub_dir, c, ip1, ip2, max_distance, max_gap, gps_table):

    df_start = pandas.DataFrame([{"latitude":numpy.nan,"longitude":numpy.nan}], index=pandas.DatetimeIndex([c["start_t"]]), columns=["latitude","longitude"])

    df_end = df_start.copy()
    df_end.index = pandas.DatetimeIndex([c["end_t"]])


    select = "timestamp, latitude, longitude"
    if c["end_t"] is None:
        # if not closed connection query all gps and close as soon as max_distance reached
        where = "timestamp >= '%s'" % c["start_t"]
    else:
        where = "timestamp BETWEEN '%s' AND '%s'" % (c["start_t"], c["end_t"])


    path1 = os.path.join(data_path, data_sub_dir,"%s_%s.db" % (c["id1"], ip1))

    gps1 = util.sqlite_to_df(path1, gps_table, select=select, where=where, index="timestamp", warn_empty=False)
    if gps1 is not None:
        if c["end_t"] is None:
            gps1 = pandas.concat([df_start,gps1])
        else:
            gps1 = pandas.concat([df_start,gps1,df_end])


    path2 = os.path.join(data_path, data_sub_dir,"%s_%s.db" % (c["id2"], ip2))

    gps2 = util.sqlite_to_df(path2, gps_table, select=select, where=where, index="timestamp", warn_empty=False)
    if gps2 is not None:
        if c["end_t"] is None:
            gps2 = pandas.concat([df_start,gps2])
        else:
            gps2 = pandas.concat([df_start,gps2,df_end])


    conn, cur = util.open_sqlite(os.path.join(data_path,"connections.db"), max_speed=True)

    if gps1 is None or gps2 is None:
        # logging.debug("Skipping conn_id %s no gps data between %s and %s" % (c["conn_id"], c["start_s"], c["end_s"]))

        if c["end_t"] is None:
            # delete open connection that can not be closed, maybe instead use a timeout
            cur.execute("DELETE FROM connections WHERE rowid=?", [c["rowid"]])
            cur.execute("UPDATE connection_events SET flag='deleted_up' WHERE timestamp=? AND conn_id=? AND state='up' AND flag='ok' AND distance IS NULL", [c["start_t"], c["conn_id"]])
            conn.commit()
        conn.close()

        return  # skip because not possible to calc any distance

    # add empty row at end of shorter one, so they have same length after resample
    if c["end_t"] is None:
        gps1_last = gps1.index[-1]
        gps2_last = gps2.index[-1]

        if gps1_last < gps2_last:
            empty = pandas.DataFrame([{"latitude":numpy.nan,"longitude":numpy.nan}], index=[gps2_last], columns=["latitude","longitude"])

            gps1 = pandas.concat([gps1,empty])
        else:
            empty = pandas.DataFrame([{"latitude":numpy.nan,"longitude":numpy.nan}], index=[gps1_last], columns=["latitude","longitude"])

            gps2 = pandas.concat([gps2,empty])


    RESAMPLE = "1s"
    gps1 = gps1.resample(rule=RESAMPLE).mean()
    gps2 = gps2.resample(rule=RESAMPLE).mean()


    if len(gps1) != len(gps2):
        logging.error("Gps data has not the same length: %s!=%s" % (len(gps1),len(gps2)))
        conn.close()
        return

    last_tick = None
    edited = False

    for i in range(0, len(gps1)):
        lat1 = gps1["latitude"][i]
        lon1 = gps1["longitude"][i]
        lat2 = gps2["latitude"][i]
        lon2 = gps2["longitude"][i]

        tick_t = gps1.index[i].to_pydatetime()

        if not numpy.isnan(lat1) and not numpy.isnan(lon1) and not numpy.isnan(lat2) and not numpy.isnan(lon2):
            d = distance.vincenty((lat1, lon1), (lat2, lon2)).meters

            if last_tick is None:
                # use current if first tick
                saved_last_tick = {"tick_t":tick_t, "d":d}
            else:
                saved_last_tick = last_tick.copy()  # need to save old for later use (fault of break in next if)
            # update value of last tick
            last_tick = {"tick_t":tick_t, "d":d}

            if d > max_distance:

                logging.debug("Found distance %s for conn_id %s at %s" % (d, c["conn_id"], tick_t))

                # If gap larger than max_gap and than d > max_distance use the last tick before gap. This avoids openening new long range connections.
                gap = (tick_t - saved_last_tick["tick_t"]).total_seconds()
                if gap > max_gap:
                    logging.debug("Using last tick after gap >%.2fmin for %s with gap end at %s" % (max_gap/60,c["conn_id"], tick_t))
                    tick_t = saved_last_tick["tick_t"]
                    d = saved_last_tick["d"]

                # insert new down event because distance is to large
                values = [tick_t, c["conn_id"], c["id1"], c["id2"], "down", d, "ok_distance_closed"]
                cur.execute("INSERT OR IGNORE INTO connection_events (timestamp, conn_id, fromId, toId, state, distance, flag) VALUES (?,?,?,?,?,?,?)", values)

                # update the old connection
                new_duration = (tick_t - c["start_t"]).total_seconds()
                values = [tick_t, new_duration, d, "distance_closed", c["rowid"]]
                cur.execute("UPDATE connections SET end_t=?, duration=?, down_distance=? ,flag=? WHERE rowid=?", values)

                edited = True  # flag if the connection was edited, used for not closed connections

                # dont need to run if is was a not closed connection
                if c["end_t"] is not None:

                    # if connection was one with to large down distance update old down event to ignore it
                    values = [c["end_t"], c["conn_id"], c["down_distance"]]
                    cur.execute("UPDATE connection_events SET flag='max_distance_down' WHERE timestamp=? AND conn_id=? AND distance=? AND flag='ok_max_distance_down'", values)

                    # fetch up events after new down event and old down event that where ignored because not_closed flag
                    values = [tick_t, c["end_t"], c["id1"], c["id2"], c["id1"], c["id2"]]
                    cur.execute("SELECT rowid, * FROM connection_events WHERE timestamp > ? AND timestamp < ? AND state='up' AND flag='not_closed' AND (fromId=? OR fromId=?) AND (toId=? OR toId=?) ORDER BY timestamp", values)
                    ups = cur.fetchall()

                    if len(ups) == 0:
                        # set old ok down event to replaced_down
                        values = [c["end_t"], c["conn_id"]]
                        cur.execute("UPDATE connection_events SET flag='replaced_down' WHERE timestamp=? AND conn_id=? AND state='down' AND flag='ok'", values)
                    elif len(ups) == 1:
                        # set found not_closed event to ok, because it is the start of a new connection
                        cur.execute("UPDATE connection_events SET flag='ok_distance_open' WHERE rowid=?", [ups[0]["rowid"]])

                        # add new connection using old down event as end, could still be to large (recursion)
                        duration = (c["end_t"] - ups[0]["timestamp"]).total_seconds()
                        values = [ups[0]["timestamp"], c["end_t"], duration, c["conn_id"], c["id1"], c["id2"], ups[0]["distance"], c["down_distance"], "new"]
                        cur.execute("INSERT OR IGNORE INTO connections (start_t, end_t, duration, conn_id, id1, id2, up_distance, down_distance, flag) VALUES (?,?,?,?,?,?,?,?,?)", values)
                        logging.warning("Unchecked connection: %s start: %s" % (c["conn_id"], ups[0]["timestamp"]))
                    elif len(ups) > 1:
                        logging.error("Not implemented, just execute again?. %s not_closed events for conn_id %s between %s - %s" % (len(ups), c["conn_id"], tick_t, c["end_t"]))

                break

    # not closed connection with all following gps distance < max_distance, use last_tick to close connection
    if c["end_t"] is None and not edited:
        if last_tick is None:  # can be None if the gps data does not overlap
            # delete that connection
            cur.execute("DELETE FROM connections WHERE rowid=?", [c["rowid"]])
            cur.execute("UPDATE connection_events SET flag='deleted_up' WHERE timestamp=? AND conn_id=? AND state='up' AND flag='ok'", [c["start_t"], c["conn_id"]])
        else:
            # insert new down event
            values = [last_tick["tick_t"], c["conn_id"], c["id1"], c["id2"], "down", last_tick["d"], "ok_gps_end"]
            cur.execute("INSERT OR IGNORE INTO connection_events (timestamp, conn_id, fromId, toId, state, distance, flag) VALUES (?,?,?,?,?,?,?)", values)

            # update the old connection
            new_duration = (last_tick["tick_t"] - c["start_t"]).total_seconds()
            values = [last_tick["tick_t"], new_duration, last_tick["d"], "gps_end", c["rowid"]]
            cur.execute("UPDATE connections SET end_t=?, duration=?, down_distance=?, flag=? WHERE rowid=?", values)

    conn.commit()
    conn.close()

def checkGPSDistance(data_path, data_sub_dir, max_distance, max_gap, gps_table):
    start_time = datetime.now()

    # read ids only once
    ids = util.readIds(os.path.join(data_path,"ids.db"))

    conn, cur = util.open_sqlite(os.path.join(data_path,"connections.db"))

    cur.execute("SELECT rowid, * FROM connections")
    connections = cur.fetchall()

    conn.close()

    pool = multiprocessing.Pool(processes=config.POOL_SIZE)

    results = []
    processes = []

    for c in connections:
        ip1 = util.getIp(ids, id=c["id1"])
        ip2 = util.getIp(ids, id=c["id2"])

        p = pool.apply_async(func=util.worker_wrapper, args=(checkGPSDistanceWorker, (data_path, data_sub_dir, dict(c), ip1, ip2, max_distance, max_gap, gps_table)))
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

    logging.info("Checked distances of %s connections in %.2f seconds" % (len(connections), (datetime.now()-start_time).total_seconds()))
###### end of connection.db ####################

####### data for paper ###################
def plotConnectionDistanceUpDown(data_path, plots_dir, filter_fromTo=None, auto_open=True):
    os.makedirs(plots_dir, exist_ok=True)

    conn, cur = util.open_sqlite(os.path.join(data_path,"connections.db"))

    distanceArray = []

    if filter_fromTo:
        where_up = "DATETIME(timestamp) BETWEEN DATETIME('%s') AND DATETIME('%s') AND state == '%s'" % (filter_fromTo[0], filter_fromTo[1], "up")
        where_conn = "DATETIME(start_t) BETWEEN DATETIME('%s') AND DATETIME('%s') OR DATETIME(end_t) > DATETIME('%s')" % (filter_fromTo[0], filter_fromTo[1], filter_fromTo[0])
        #where_conn = "DATETIME(start_s) BETWEEN DATETIME('%s') AND DATETIME('%s') OR DATETIME(end_s) > DATETIME('%s')" % (filter_fromTo[0], filter_fromTo[1], filter_fromTo[0])
    else:
        where_up = None
        where_conn = None

#    df_events_up = util.sqlite_to_df(conn, "connection_events", order="epoch", index="epoch", where=where_up)
    df_events_up = util.sqlite_to_df(conn, "connection_events", order="timestamp", index="timestamp", where=where_up)
    df_connections = util.sqlite_to_df(conn, "connections", parse_s=["start_t", "end_t"], where=where_conn)

    conn.close()

        # Events Time Series ###
    RESAMPLE = "2min"

    connectionOverTimeX = []
    connectionOverTimeY = []

    for flag in df_events_up["flag"].unique():
        print ("flag: %s" % flag)
        if flag == "ok":
            df_flag = df_events_up[df_events_up["flag"] == flag]
            #df_flag['timestamp'] = pandas.to_datetime(df_flag['timestamp'])
            #df_flag = df_flag.set_index('timestamp')
            #print(list(df_flag.columns.values))
            #print ("Checking: %s\r\n" % df_flag)
            df_flag = df_flag.resample(rule=RESAMPLE).count()
            connectionOverTimeX.append(df_flag.index)
            connectionOverTimeY.append(df_flag["flag"])

    # Connection Distance Histo Ok ###
    up_distance = df_connections["up_distance"][df_connections["up_distance"].notnull()]
    down_distance = df_connections["down_distance"][df_connections["down_distance"].notnull()]
    #up_distance = df_connections["up_distance"][df_connections["up_distance"]]
    #down_distance = df_connections["down_distance"][df_connections["down_distance"]]

    avg_up_distance = up_distance.mean()
    avg_down_distance = down_distance.mean()

    max_d = 500  # everything over 500m goes into one bucket

    scaled_up = sorted([d if d <= max_d else max_d + 0.9 for d in up_distance])
    scaled_down = sorted([d if d <= max_d else max_d + 0.9 for d in down_distance])

    distanceArray.append(scaled_up)    

    return distanceArray, connectionOverTimeX, connectionOverTimeY

def plotConnectionDurationUpDown(data_path, plots_dir, fromTo, filter_fromTo=None, auto_open=True, connections_compare_reports=None, compare_names=None):

    conn, cur = util.open_sqlite(os.path.join(data_path,"connections.db"))

    if filter_fromTo:
        where = " WHERE DATETIME(start_t) BETWEEN DATETIME('%s') AND DATETIME('%s') OR DATETIME(end_t) > DATETIME('%s')" % (filter_fromTo[0], filter_fromTo[1], filter_fromTo[0])
    else:
        where = ""

    connections = util.query_sqlite(cur, "SELECT * FROM connections%s" % where)
    connections = [dict(c) for c in connections]  # convert to dict for pickle
    conn.close()
    if connections is None:
        logging.error("Could not load connections from database")
        return

    work_connections = [connections]

    durations = []
    for connections in work_connections:
        durations.append([int((c["end_t"]-c["start_t"]).total_seconds()) for c in connections])
#        durations.append([int((c["end_t"]-c["start_t"])) for c in connections])

    return durations

def plotTogether(data_path, plots_dir, fromTo, filter_fromTo=None, auto_open=True, connections_compare_reports=None, compare_names=None):
    duration = plotConnectionDurationUpDown(data_path, plots_dir, fromTo, filter_fromTo, auto_open, connections_compare_reports, compare_names)
    print ("plots_dir %s" % plots_dir)
    print ("data_path %s" % data_path)
    print ("filter_fromTo %s" % filter_fromTo)
    print ("auto_open %s" % auto_open)
    distance, timeX, timeY = plotConnectionDistanceUpDown(data_path, plots_dir, filter_fromTo, auto_open)


    clusterX, clusterY = plotClusteringCoefficient(config.DATA_PATH, config.PLOTS_DIR, config.WALKING_FROM_TO, filter_fromTo=config.WALKING_FROM_TO, auto_open=config.AUTO_OPEN_PLOTS, connections_compare_reports=config.CONNECTIONS_COMPARE_REPORTS, compare_names=config.CONNECTIONS_COMPARE_NAMES)


    statistics_functions.ecdfUsingMatplot(distance, 230, 'Distance in m', 'ECDF P(d<d1)', 200)
    statistics_functions.ecdfUsingMatplot(duration, 10000, 'Contact duration in s', 'ECDF P(t<t1)', 10000, isLog = True)
    statistics_functions.timeUsingMatplotUnique(timeX, timeY, 'time', 'count')
    
    print("#### Plot DATA ###")
    print("### X - BANANA Data ###")
    for i in clusterX:
        print("%s" % i)
    print("### Y - BANANA Data ###")
    for i in clusterY:
        print("%f" % i)
    
    print("x1: %s" % clusterX)
    print("y1: %s" % clusterY)
    print ("#### END Plot DATA ###")

    statistics_functions.timeUsingMatplotUnique(clusterX, clusterY, 'time', 'count')
    #statistics_functions.timeUsingMatplot(distance, timeX)

######################## END OF CHANGES ###################

def computeClusteringCoefficients(intervalls, nodes, connections):
    cc_s = []
    mean_cc_s = []

    for i in intervalls:
        date = i.to_pydatetime()

        i_connections = [c for c in connections if date >= c["start_t"] and date <= c["end_t"]]
        edges = [(c["id1"], c["id2"]) for c in i_connections]

        G = networkx.Graph()
        G.add_nodes_from(nodes)
        G.add_edges_from(edges)

        clustering_coefficients = networkx.clustering(G)
        cc_s.append(clustering_coefficients)

        mean_cc = sum(clustering_coefficients.values())/len(clustering_coefficients.keys())
        mean_cc_s.append(mean_cc)

    return {"all":cc_s, "mean":mean_cc_s}

def plotClusteringCoefficient(data_path, plots_dir, fromTo, filter_fromTo=None, auto_open=True, connections_compare_reports=None, compare_names=None):

    RESAMPLE = "10s"

    conn, cur = util.open_sqlite(os.path.join(data_path,"connections.db"))

    if filter_fromTo:
        where = " WHERE DATETIME(start_t) BETWEEN DATETIME('%s') AND DATETIME('%s') OR DATETIME(end_t) > DATETIME('%s')" % (filter_fromTo[0], filter_fromTo[1], filter_fromTo[0])
    else:
        where = ""

    connections = util.query_sqlite(cur, "SELECT * FROM connections%s" % where)
    conn.close()
    if connections is None:
        logging.error("Could not load connections from database")
        return

    conn, cur = util.open_sqlite(os.path.join(data_path,"ids.db"))

    ids = util.query_sqlite(cur, "SELECT DISTINCT id FROM ids")
    conn.close()
    if ids is None:
        logging.error("Could not load node ids from database")
        return
    nodes = set([i["id"] for i in ids])

    if filter_fromTo:
        intervall_start = datetime.strptime(filter_fromTo[0], "%Y-%m-%d %H:%M:%S")
        intervall_end = datetime.strptime(filter_fromTo[1], "%Y-%m-%d %H:%M:%S")
    else:
        intervall_start = datetime.strptime(fromTo[0], "%Y-%m-%d %H:%M:%S")
        intervall_end = datetime.strptime(fromTo[1], "%Y-%m-%d %H:%M:%S")

    intervalls = pandas.date_range(start=intervall_start, end=intervall_end, freq=RESAMPLE)

    smarter_cc = computeClusteringCoefficients(intervalls, nodes, connections)

    return intervalls, smarter_cc["mean"]

if __name__ == "__main__":

    gps_table = "gps_data"

    # # filters & combines connection log events into connections.db with raw_connections, connection_events & connections tables
    createOneConnections(config.DATA_PATH, config.DATA_SUB_DIR, config.MAX_CONN_DISTANCE, config.CONN_LOOKUP_OFFSET, config.GPS_LOOKUP_OFFSET, gps_table)

    # # modifies connections tables, mainly removes connections with to large distances
    checkGPSDistance(config.DATA_PATH, config.DATA_SUB_DIR, config.MAX_CONN_DISTANCE, config.MAX_END_GPS_GAP, gps_table)

    # print data for the paper
    plotTogether(config.DATA_PATH, config.PLOTS_DIR, config.FROM_TO, filter_fromTo=config.WALKING_FROM_TO, auto_open=config.AUTO_OPEN_PLOTS, connections_compare_reports=config.CONNECTIONS_COMPARE_REPORTS, compare_names=config.CONNECTIONS_COMPARE_NAMES)
   

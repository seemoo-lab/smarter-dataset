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
import warnings
import logging
import json
from datetime import datetime
from operator import itemgetter
import hashlib

import pandas
from natsort import natsorted

import config
import util

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from plotly.offline import plot
    import plotly.graph_objs as go
    from plotly.colors import DEFAULT_PLOTLY_COLORS


def realmFuseWorker(file, ids):
    conn, cur = util.open_sqlite(file)

    ip = util.extractIp(file)
    if ip is None:
        logging.error("Could not find Ip in %s" % file)
        return

    _id = util.getId(ids, ip=ip)
    if _id is None:
        logging.error("Could not find Id for ip %s" % ip)
        return


    realm_data = util.query_sqlite(cur, "SELECT * FROM realm_data", if_no_table=[])
    if len(realm_data) == 0:
        logging.warning("%s has no realm_data" % os.path.basename(file))
        return

    realm_data = [dict(i) for i in realm_data]  # cast to dict for get() function

    inserts = []

    for row in realm_data:

        received = row["isReceived"]
        if received != row["isDelieverd"]:  # isReceived is sometimes 1 while isDelieverd is 0
            logging.error("isReceived: %s != isDelieverd: %s for %s at %s from %s" % (row["isReceived"], row["isDelieverd"], row["messageType"], row["index"], os.path.basename(file)))
            continue


        other_id = None
        messageType = None
        lat = row.get("j_latitude",None)
        lon = row.get("j_longitude",None)
        category = None
        text1, text2, text3 = (None, None, None)


        if row["messageType"] is not None:
            messageType = row["messageType"].replace("_message","")

        if messageType == "hilferuf":
            if received == 1:
                other_id = util.getId(ids, dtn_id=row["dtn_id"])
                if other_id is None:
                    logging.error("Could not find id for %s in %s at %s" % (row["dtn_id"], os.path.basename(file), row["index"]))
                    continue
            else:
                other_id = None  # if send its a broadcast

            text1 = row["j_passiertText"]
            text2 = row["j_beschreibenText"]
            text3 = row["j_isVerletzte"]

            if row["j_categories"] is not None:
                category = {}
                category_checks = json.loads(row["j_categories"])
                rettungsdienst = []
                polizei = []
                netzwerk = []

                for key in category_checks:
                    i = category_checks[key]
                    if i["isChecked"]:
                        if i["parentCategory"] == "rettungsdiens":  # sic
                            rettungsdienst.append(i["title"])
                        elif i["parentCategory"] == "smarternetCategory":
                            netzwerk.append(i["title"])
                        elif i["parentCategory"] == "polizei":
                            polizei.append(i["title"])
                        else:
                            logging.error("Unkown parentCategory %s from %s at %s" % (i["parentCategory"], os.path.basename(file), row["index"]))

                category["Rettungsdienst"] = rettungsdienst
                category["Polizei"] = polizei
                category["Netzwerk"] = netzwerk
                category = json.dumps(category)

            hash_content = "%s%s%s%s%s" % (messageType, category, text1, text2, text3)

        elif messageType == "ressourcenmarkt":
            # seams to be always 0, no way to know if this message send by this node or received
            if received != 0:
                logging.error("ressourcenmarkt with received !=0 from %s at %s" % (os.path.basename(file), row["index"]))

            # dtn_id and j_from differ from nodes own dtn_ids so probably received

            search_id = util.getId(ids, dtn_id=row["dtn_id"])
            if search_id is None:
                logging.error("Could not find id for %s in %s at %s" % (row["dtn_id"], os.path.basename(file), row["index"]))
                continue

            if search_id == _id:  # from is own id then this was probably send into the network, but could also be duplicate receive (or does the bundle layer filter these out before)
                received = 0
            else:
                received = 1

            text1 = row["j_title"]
            text2 = row["j_desc"]
            text3 = row["j_Quanitity"]  # sic
            category = row["j_category"]

            hash_content = "%s%s%s%s%s" % (messageType, category, text1, text2, text3)
            # for 0b3e0292e03a9d67fcdfb8d6658e58a8 (#481) all None (realm_creator aslo None) but they clearly dont belong together, no way to generate id
            # ccf5d56fb962485f4d9148072d17b51e multiple with received 0 but equal realm_creator one of them has status -1 ???
            # cfab094f0ea57d155ff1508ac90a0921 multiple with received 0 and equal realm_creator

        elif messageType == "ressourcenmarkt_delete":
            # has nearly no data, check if received is set properly, seams to be always 0 -> no way to differ direction
            if received != 0:
                logging.error("ressourcenmarkt_delete with received !=0 from %s at %s" % (os.path.basename(file), row["index"]))

            # hash_content = "%s%s%s" % (messageType, _id, row["timestamp"]) # not usefull unable to detect if the same delte arrives somewhere
            hash_content = None

        elif messageType == "chat":
            if row["j_from"] is None or row["j_to"] is None or row["dtn_id"] is None:
                logging.error("One of the chat message ids is None from %s at %s" % (os.path.basename(file), row["index"]))
                continue

            from_id = util.getId(ids, dtn_id=row["j_from"])
            if from_id is None:
                logging.error("Could not find from_id for %s in %s at %s" % (row["j_from"], os.path.basename(file), row["index"]))

            to_id = util.getId(ids, dtn_id=row["j_to"])
            if to_id is None:
                logging.error("Could not find to_id for %s in %s at %s" % (row["j_to"], os.path.basename(file), row["index"]))

            other_id = util.getId(ids, dtn_id=row["dtn_id"])
            if other_id is None:
                logging.error("Could not find other_id for %s in %s at %s" % (row["dtn_id"], os.path.basename(file), row["index"]))

            if received == 0:
                if not (_id == from_id and other_id == to_id):
                    logging.error("Ids of sending chat message dont match from %s at %s" % (os.path.basename(file), row["index"]))
                    continue
            else:
                if not (_id == to_id and other_id == from_id):
                    logging.error("Ids of receiving chat message dont match from %s at %s" % (os.path.basename(file), row["index"]))
                    continue

            text1 = row["j_message"]

            if _id is None or other_id is None:
                logging.error("Could not find all chat message ids, from %s at %s" % (os.path.basename(file), row["index"]))
                continue

            ordered_ids = sorted([_id, other_id])
            hash_content = "%s%s%s%s" % (messageType, ordered_ids[0], ordered_ids[1], text1)
            # unwanted collisions if the same text is send again, happens ~20 times

        elif messageType == "lebenszeichen" or messageType == "personenfinder":
            other_id = util.getId(ids, dtn_id=row["dtn_id"])
            if other_id is None:
                logging.error("Could not find id for %s in %s at %s" % (row["dtn_id"], os.path.basename(file), row["index"]))
                continue

            if received == 1:
                hash_content = "%s%s->%s" % (messageType, other_id, _id)
            else:
                hash_content = "%s%s->%s" % (messageType, _id, other_id)
            # many collisions when received=0, double sends?

        else:
            logging.error("Unkown messageType %s from %s at %s" % (row["messageType"], os.path.basename(file), row["index"]))
            continue


        # md5 hash as message id
        if hash_content is not None:
            md5 = hashlib.md5(hash_content.encode("utf-8")).hexdigest()
        else:
            md5 = None  # for some types it is not possible to generate a meaningful msg id


        inserts.append([row["timestamp"], row.get("j_sentTime",None), _id, other_id, received, messageType, md5, row["json_size"], lat, lon, category, text1, text2, text3, row.get("j_creatorORMMessageId",None),row.get("status",None)])


    if len(inserts) > 0:
        return inserts
    else:
        return


def realmFuse(input_dir, data_path):

    ids = util.readIds(os.path.join(data_path,"ids.db"))

    results = util.forAll(input_dir, ".db", realmFuseWorker, (ids,))
    # flatten results
    inserts = [u for r in results for u in r]

    conn, cur = util.open_sqlite(os.path.join(data_path,"realm.db"), create=True, max_speed=True)

    cur.execute("DROP TABLE IF EXISTS realm_messages")
    cur.execute("CREATE TABLE realm_messages (timestamp TIMESTAMP NOT NULL, sent_time TIMESTAMP, id INTEGER NOT NULL, other_id INTEGER, received INTEGER, type TEXT NOT NULL, msg_id TEXT, size INTEGER, latitude REAL, longitude REAL, category TEXT, text1 TEXT, text2 TEXT, text3 TEXT, realm_creator TEXT, status INTEGER, PRIMARY KEY (timestamp, sent_time, id, other_id, type, msg_id))")

    cur.executemany("INSERT INTO realm_messages (timestamp,sent_time,id,other_id,received,type,msg_id,size,latitude,longitude,category,text1,text2,text3,realm_creator,status) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", inserts)

    conn.commit()
    conn.close()

    logging.info("Inserted %s realm messages" % len(inserts))


def plotRealmMessagesWorker(file):

    realm_gps = util.sqlite_to_df(file, "realm_data", select="timestamp, j_latitude, j_longitude", where="j_latitude > 0 AND j_longitude > 0", index="timestamp")
    if realm_gps is None:
        return

    realm_gps.sort_index(inplace=True)

    # have to use data for return because Scattermapbox throws error with multiprocessing
    return {"name":"%s" % os.path.basename(file),"lat":realm_gps["j_latitude"].values, "lon":realm_gps["j_longitude"].values}


def plotRealmMessages(input_dir, plots_dir, only=None, auto_open=True):

    os.makedirs(plots_dir, exist_ok=True)

    traces = []

    if only:
        data = plotRealmMessagesWorker(os.path.join(input_dir, only))

        title = "%s Realm Messages of %s" % (len(data), only)
        filename = "realm_gps_%s.html" % only
    else:
        data = util.forAllMP(input_dir, ".db", plotRealmMessagesWorker)

        messages = 0
        for d in data:
            messages += len(d["lat"])

        title = "%s Realm Messages of %s files" % (messages, len(data))
        filename = "realm_gps.html"


    data = natsorted(data, key=itemgetter("name"))

    for d in data:
        traces.append(go.Scattermapbox(
                name=d["name"],
                mode="markers",
                connectgaps=False,
                lat=d["lat"],
                lon=d["lon"],
                opacity=0.7,
                line=go.Line(width=0.5, color="#888"),
                marker=go.Marker(color="rgb(255, 0, 0)"),
                hoverinfo="lat+lon+name"
                ))

    layout = go.Layout(title=title, titlefont=dict(size=16), showlegend=True, mapbox=dict(accesstoken=config.MAPBOX_TOKEN, zoom=8, center=dict(lat=51.81932, lon=8.77124)))

    layout["updatemenus"] = [dict(buttons=[dict(label="show all", method="restyle", args=["visible", True]), dict(label="hide all", method="restyle", args=["visible", "legendonly"])])]

    fig = go.Figure(data=traces, layout=layout)

    plot(fig, filename=os.path.join(plots_dir, filename), show_link=False, auto_open=auto_open)


def plotRealmMessageStatsWorker(file):

    realm_data = util.sqlite_to_df(file, "realm_data", select="timestamp, messageType, json_size", index="timestamp")
    if realm_data is None:
        return

    realm_data["source"] = os.path.basename(file)

    return realm_data


def plotRealmMessageStats(data_path, plots_dir, fromTo, filter_fromTo=None, auto_open=True):

    os.makedirs(plots_dir, exist_ok=True)

    conn, cur = util.open_sqlite(os.path.join(data_path, "realm.db"))
    realm_messages = cur.execute("SELECT * FROM realm_messages ORDER BY timestamp")
    realm_messages = [dict(i) for i in cur.fetchall()]

    if filter_fromTo:
        realm_messages_filtered = cur.execute("SELECT * FROM realm_messages WHERE DATETIME(timestamp) BETWEEN DATETIME('%s') AND DATETIME('%s') ORDER BY timestamp" % (filter_fromTo[0], filter_fromTo[1]))
        realm_messages_filtered = [dict(i) for i in cur.fetchall()]
    else:
        realm_messages_filtered = realm_messages

    conn.close()

    # b,o,g,r,l,b
    message_types = ["hilferuf","personenfinder","lebenszeichen","ressourcenmarkt","chat","ressourcenmarkt_delete"]

    # make color always the same for type
    colors_type = {}
    for i,t in enumerate(message_types):
        colors_type[t] = DEFAULT_PLOTLY_COLORS[i]

    sim_start = datetime.strptime(fromTo[0], "%Y-%m-%d %H:%M:%S")
    sim_end = datetime.strptime(fromTo[1], "%Y-%m-%d %H:%M:%S")

    # Time Series ###
    RESAMPLE = "1min"

    traces = []

    for messageType in message_types:
        realm_messageType = [i for i in realm_messages if i["type"] == messageType and i["received"] == 0]

        df_messageType = pandas.DataFrame(realm_messageType, index=[i["timestamp"] for i in realm_messageType], columns=realm_messageType[0].keys())
        df_messageType = df_messageType.resample(rule=RESAMPLE).count()

        traces.append(go.Scatter(x=df_messageType.index, y=df_messageType["type"], name=messageType, line=dict(color=colors_type[messageType])))

    layout = go.Layout(title="Time Series of %s Realm Messages (received = 0)" % len([i for i in realm_messages if i["received"] == 0]),
                       yaxis=dict(title="Count"),
                       titlefont=dict(size=21),
                       xaxis=dict(rangeselector=dict(buttons=list([
                                                                    dict(count=30,
                                                                         label='30min',
                                                                         step='minute',
                                                                         stepmode='backward'),
                                                                    dict(count=1,
                                                                         label='1h',
                                                                         step='hour',
                                                                         stepmode='backward'),
                                                                    dict(count=4,
                                                                         label='4h',
                                                                         step='hour',
                                                                         stepmode='backward'),
                                                                    dict(step='all')
                                                                ])
                                                     ),
                                  rangeslider=dict(),
                                  type='date')
                       )
    fig = go.Figure(data=traces, layout=layout)
    plot(fig, filename=os.path.join(plots_dir,"realm_timeseries.html"), show_link=False, auto_open=auto_open)
    ###

    # Time Series ###
    RESAMPLE = "1min"

    traces = []

    for messageType in message_types:
        realm_messageType = [i for i in realm_messages if i["type"] == messageType and i["received"] == 1]

        if len(realm_messageType) == 0:
            logging.info("No messages with received = 1 for %s" % messageType)
            continue

        df_messageType = pandas.DataFrame(realm_messageType, index=[i["timestamp"] for i in realm_messageType], columns=realm_messageType[0].keys())
        df_messageType = df_messageType.resample(rule=RESAMPLE).count()

        traces.append(go.Scatter(x=df_messageType.index, y=df_messageType["type"], name=messageType, line=dict(color=colors_type[messageType])))

    layout = go.Layout(title="Time Series of %s Realm Messages (received = 1)" % len([i for i in realm_messages if i["received"] == 1]),
                       yaxis=dict(title="Count"),
                       titlefont=dict(size=21),
                       xaxis=dict(rangeselector=dict(buttons=list([
                                                                    dict(count=30,
                                                                         label='30min',
                                                                         step='minute',
                                                                         stepmode='backward'),
                                                                    dict(count=1,
                                                                         label='1h',
                                                                         step='hour',
                                                                         stepmode='backward'),
                                                                    dict(count=4,
                                                                         label='4h',
                                                                         step='hour',
                                                                         stepmode='backward'),
                                                                    dict(step='all')
                                                                ])
                                                     ),
                                  rangeslider=dict(),
                                  type='date')
                       )
    fig = go.Figure(data=traces, layout=layout)
    plot(fig, filename=os.path.join(plots_dir,"realm_timeseries_received.html"), show_link=False, auto_open=auto_open)
    ###

    # Type Pie Chart ###
    values = []
    value_colors = []
    for label in message_types:
        values.append(len([i for i in realm_messages_filtered if i["type"] == label]))
        value_colors.append(colors_type[label])

    layout = go.Layout(title="<b>Type Distribution of %s Messages</b>" % len(realm_messages), titlefont=dict(size=27))

    trace = go.Pie(showlegend=False, labels=list(message_types), values=values, textinfo="value+percent+label", direction="clockwise", rotation=-90, textfont=dict(size=35, color="black"), opacity=0.75, marker=dict(colors=value_colors))

    fig = go.Figure(data=[trace], layout=layout)

    plot(fig, filename=os.path.join(plots_dir,"realm_pie.html"), show_link=False, auto_open=auto_open)
    ###

    # Type Pie Chart with unique msg ids ###
    values = []
    value_colors = []
    labels = []
    for label in message_types:
        if label != "ressourcenmarkt_delete":  # has no msg_id
            msg_ids = [i["msg_id"] for i in realm_messages_filtered if i["type"] == label]
            values.append(len(set(msg_ids)))
            value_colors.append(colors_type[label])
            labels.append(label)

    layout = go.Layout(title="<b>Type Distribution of %s unique Messages</b>" % sum(values), titlefont=dict(size=27))

    trace = go.Pie(showlegend=False, labels=labels, values=values, textinfo="value+percent+label", direction="clockwise", rotation=-90, textfont=dict(size=35, color="black"), opacity=0.75, marker=dict(colors=value_colors))

    fig = go.Figure(data=[trace], layout=layout)

    plot(fig, filename=os.path.join(plots_dir,"realm_pie_msg_ids.html"), show_link=False, auto_open=auto_open)
    ###

    # Stacked size histo ###
    got_msg_ids = set()
    unique_messages = []
    for i in realm_messages_filtered:
        if i["msg_id"] not in got_msg_ids:
            unique_messages.append(i)
            got_msg_ids.add(i["msg_id"])

    traces = []
    for label in message_types:
        if label != "ressourcenmarkt_delete":  # has no size
            x = [i["size"] for i in unique_messages if i["type"] == label]
            traces.append(go.Histogram(name=label, x=x, opacity=0.75, marker=dict(color=colors_type[label])))

            avg_x = sum(x)/len(x)
            min_x = min(x)
            max_x = max(x)

            logging.info("%s: avg: %.2f, min: %.0f, max: %.0f" % (label, avg_x, min_x, max_x))

    sizes = [i["size"] for i in unique_messages if i["size"] is not None]
    avg_size = sum(sizes)/len(sizes)
    min_size = min(sizes)
    max_size = max(sizes)

    logging.info("Max size of %s came from id %s" % (max_size, [i for i in unique_messages if i["size"] == max_size][0]["id"]))

    layout = go.Layout(title="<b>Sizes of %s Realm Messages</b><br>avg: %.2f, min: %.2f, max: %.2f" % (len(unique_messages), avg_size, min_size, max_size), titlefont=dict(size=25), xaxis=dict(title="Size in Byte", titlefont=dict(size=22), tickfont=dict(size=24)), yaxis=dict(title="# Messages", titlefont=dict(size=22), tickfont=dict(size=24)), bargap=0.1, barmode="stack", legend=dict(font=dict(size=21)))

    fig = go.Figure(data=traces, layout=layout)

    plot(fig, filename=os.path.join(plots_dir,"realm_sizes.html"), show_link=False, auto_open=auto_open)
    ###

    # Message Spread ###
    labels = ["hilferuf","ressourcenmarkt"]
    node_ids = list(set([i["id"] for i in realm_messages_filtered]))
    traces = []
    receive_counts = {}

    for label in labels:
        label_messages = [i for i in realm_messages_filtered if i["type"] == label]
        label_msg_ids = list(set([i["msg_id"] for i in label_messages]))


        for msg_id in label_msg_ids:
            received = [i for i in label_messages if i["msg_id"] == msg_id and i["received"] == 1]

            label_counts = receive_counts.get(label, [])

            label_counts.append(len(set([i["id"] for i in received])))  # exclude double received

            receive_counts[label] = label_counts

        traces.append(go.Histogram(name=label, x=receive_counts[label], opacity=0.75, autobinx=False, xbins=dict(start=0,end=len(node_ids),size=1), marker=dict(color=colors_type[label])))

    num_messages = 0
    subtitle = "avg. spread ratio - "
    for label in labels:
        ratios = []
        for c in receive_counts[label]:
            ratios.append(c/len(node_ids))
        avg = sum(ratios)/len(ratios)
        num_messages += len(ratios)
        subtitle += "%s: %.2f (#%s), " % (label, avg, len(ratios))
    subtitle = subtitle[:-2]


    layout = go.Layout(title="<b>Message Spread (%s Nodes, %s Messages)</b><br>%s" % (len(node_ids), num_messages, subtitle), titlefont=dict(size=27), xaxis=dict(title="Number of Nodes that received message", titlefont=dict(size=24), tickfont=dict(size=24)), yaxis=dict(title="Number of Messages", titlefont=dict(size=24), tickfont=dict(size=24)), bargap=0.1, barmode="stack", legend=dict(font=dict(size=24)))

    fig = go.Figure(data=traces, layout=layout)

    plot(fig, filename=os.path.join(plots_dir,"realm_message_spread.html"), show_link=False, auto_open=auto_open)
    ###

    # Delays (with sent_time) Histo ###
    traces = []
    delays = {}

    for label in message_types:
        label_messages = [i for i in realm_messages_filtered if i["type"] == label and i["received"] == 1]

        for m in label_messages:

            if m["msg_id"] == "0b3e0292e03a9d67fcdfb8d6658e58a8":
                # logging to make aware that this happens relatively often (~450)
                # sadly ressourcenmarkt msg with content dont have a sent_time -> all ressourcenmarkt msg will get irgnored
                logging.warning("Skipped malformed ressourcenmarkt msg with all None fields")
                continue

            sent_t = m.get("sent_time", None)

            if sent_t is not None:
                if sent_t < sim_start or sent_t > sim_end:
                    logging.warning("Ignored %s message sent_t %s for msg_id %s at %s not in simulation time" % (label, sent_t, m["msg_id"], m["id"]))
                    continue

                label_delays = delays.get(label,[])

                delay = (m["timestamp"] - sent_t).total_seconds()

                if delay < 0:
                    logging.error("Ignored negative delay of %s for %s msg_id %s at %s" % (delay, label, m["msg_id"], m["id"]))
                    continue

                delay = delay/60

                label_delays.append(delay)

                delays[label] = label_delays

        if delays.get(label, False):
            traces.append(go.Histogram(name=label, x=delays[label], opacity=0.75, autobinx=False, xbins=dict(start=0, end=121, size=1), marker=dict(color=colors_type[label])))

    num_messages = 0
    subtitle = ""
    for label in message_types:
        if delays.get(label, False):
            avg = sum(delays[label])/len(delays[label])
            num_messages += len(delays[label])
            subtitle += "avg %s: %.2f (#%s), " % (label, avg, len(delays[label]))
    subtitle = subtitle[:-2]


    layout = go.Layout(title="<b>Message Delays of %s Messages</b><br>%s" % (num_messages, subtitle), titlefont=dict(size=24), xaxis=dict(title="Delay in minutes", titlefont=dict(size=22), tickfont=dict(size=22)), yaxis=dict(title="#Messages", titlefont=dict(size=22), tickfont=dict(size=22)), bargap=0.1, barmode="stack", legend=dict(font=dict(size=24)))

    fig = go.Figure(data=traces, layout=layout)

    plot(fig, filename=os.path.join(plots_dir,"realm_message_delays.html"), show_link=False, auto_open=auto_open)
    ###

    # Delays (with received) Histo ###
    traces = []
    delays = {}

    for label in message_types:
        send_ids = set([i["msg_id"] for i in realm_messages_filtered if i["type"] == label and i["received"] == 0])
        if len([i for i in realm_messages_filtered if i["type"] == label and i["received"] == 1]) == 0:
            logging.info("Skipped delay calculation for %s, because received is always 0" % label)
            continue

        for msg_id in send_ids:
            sends = [i for i in realm_messages_filtered if i["msg_id"] == msg_id and i["received"] == 0]

            sent_t = None
            if len(sends) == 0:
                continue  # can not detect sent_t
            if len(sends) == 1:
                sent_t = sends[0]["timestamp"]
            elif len(sends) > 1:  # should only be 1 or 0, some collisions on ressourcenmarkt with same content
                logging.error("Ignored %s %s because %s received=0 entries exist" % (msg_id, label, len(sends)))

            if sent_t is not None:
                receives = [i for i in realm_messages_filtered if i["msg_id"] == msg_id and i["received"] == 1]

                for m in receives:
                    label_delays = delays.get(label,[])

                    delay = (m["timestamp"] - sent_t).total_seconds()

                    if delay < 0:
                        logging.error("Ignored negative delay of %s for %s msg_id %s at %s" % (delay, label, m["msg_id"], m["id"]))
                        continue

                    delay = delay/60

                    label_delays.append(delay)

                    delays[label] = label_delays

        if delays.get(label, False):
            traces.append(go.Histogram(name=label, x=delays[label], opacity=0.75, autobinx=False, xbins=dict(start=0, end=121, size=1), marker=dict(color=colors_type[label])))

    num_messages = 0
    subtitle = ""
    for label in message_types:
        if delays.get(label, False):
            avg = sum(delays[label])/len(delays[label])
            num_messages += len(delays[label])
            subtitle += "%s: %.2f (#%s), " % (label, avg, len(delays[label]))
    subtitle = subtitle[:-2]


    layout = go.Layout(title="<b>Message Delays of %s Messages</b><br>avg. delay - %s" % (num_messages, subtitle), titlefont=dict(size=27), xaxis=dict(title="Delay in minutes", titlefont=dict(size=24), tickfont=dict(size=24)), yaxis=dict(title="Number of Messages", titlefont=dict(size=24), tickfont=dict(size=24)), bargap=0.1, barmode="stack", legend=dict(font=dict(size=24)))

    fig = go.Figure(data=traces, layout=layout)

    plot(fig, filename=os.path.join(plots_dir,"realm_message_delays_2.html"), show_link=False, auto_open=auto_open)
    ###


if __name__ == "__main__":

    # fuse all realm data into one sqlite database & clean up some fields
    realmFuse(config.DATA_DIR, config.DATA_PATH)

    # plot all realm messages with gps coordinates
    plotRealmMessages(config.DATA_DIR, config.PLOTS_DIR, auto_open=config.AUTO_OPEN_PLOTS)

    # plot some stats about the realm data
    plotRealmMessageStats(config.DATA_PATH, config.PLOTS_DIR, config.FROM_TO, filter_fromTo=config.FROM_TO, auto_open=config.AUTO_OPEN_PLOTS)

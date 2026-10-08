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

# ignore plotly's "Looks like you don't have 'read-write' permission to your 'home' directory" warning which happens during import in multiprocessing workers
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

if __name__ == "__main__":

    # fuse all realm data into one sqlite database & clean up some fields
    realmFuse(config.DATA_DIR, config.DATA_PATH)

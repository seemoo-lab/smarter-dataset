#!/usr/bin/env python3
import os
from datetime import datetime
import logging

import config
import util


def fuseDatabasesWorker(file):
    conn, cur = util.open_sqlite(file)

    info = util.query_sqlite(cur, "SELECT * FROM device_information")
    if info is not None:
        info = dict(info[0])

    gps_data = [dict(i) for i in util.query_sqlite(cur, "SELECT * FROM gps_data", if_no_table=[])]
    smoothed_gps_data = [dict(i) for i in util.query_sqlite(cur, "SELECT * FROM smoothed_gps_data", if_no_table=[])]
    sensor_data = [dict(i) for i in util.query_sqlite(cur, "SELECT * FROM sensor_data", if_no_table=[])]

    if info is None and not gps_data and not smoothed_gps_data:
        return None  # just a dummy db

    return {"info":info, "gps_data":gps_data, "smoothed_gps_data": smoothed_gps_data, "sensor_data": sensor_data}


def fuseDatabases(data_path, sub_dir, output_file):
    start_time = datetime.now()

    conn, cur = util.open_sqlite(output_file, create=True, max_speed=True)

    cur.execute("DROP TABLE IF EXISTS options")
    cur.execute("CREATE TABLE options (key TEXT PRIMARY KEY, value TEXT)")

    cur.execute("DROP TABLE IF EXISTS nodes")
    cur.execute("CREATE TABLE nodes (id INTEGER PRIMARY KEY, ip TEXT NOT NULL, imei TEXT, brand TEXT, model TEXT)")

    cur.execute("DROP TABLE IF EXISTS ids")
    cur.execute("CREATE TABLE ids (id INTEGER, ip TEXT, dtn_id TEXT UNIQUE, source TEXT NOT NULL, PRIMARY KEY (id, ip, dtn_id))")

    cur.execute("DROP TABLE IF EXISTS movements")
    cur.execute("CREATE TABLE movements (timestamp TIMESTAMP NOT NULL, id INTEGER NOT NULL, latitude REAL NOT NULL, longitude REAL NOT NULL, altitude REAL, accuracy REAL, distance REAL, speed REAL)")
    cur.execute("CREATE INDEX movements_timestamp_index ON movements(timestamp)")
    cur.execute("CREATE INDEX movements_id_index ON movements(id)")

    cur.execute("DROP TABLE IF EXISTS smoothed_movements")
    cur.execute("CREATE TABLE smoothed_movements (timestamp TIMESTAMP NOT NULL, id INTEGER NOT NULL, latitude REAL NOT NULL, longitude REAL NOT NULL, distance REAL, speed REAL)")
    cur.execute("CREATE INDEX smoothed_movements_timestamp_index ON smoothed_movements(timestamp)")
    cur.execute("CREATE INDEX smoothed_movements_id_index ON smoothed_movements(id)")

    # neighbors.db, has to be created dynamically

    cur.execute("DROP TABLE IF EXISTS connections")
    cur.execute("CREATE TABLE connections (start_t TIMESTAMP, end_t TIMESTAMP, duration REAL, conn_id INTEGER NOT NULL, id1 INTEGER, id2 INTEGER, up_distance REAL, down_distance REAL, flag TEXT, PRIMARY KEY (start_t, end_t, conn_id))")

    cur.execute("DROP TABLE IF EXISTS bundle_events")
    cur.execute("CREATE TABLE bundle_events (timestamp TIMESTAMP NOT NULL, bundle_id TEXT NOT NULL, id INTEGER NOT NULL, other_id INTEGER, event TEXT NOT NULL, event_reason TEXT, PRIMARY KEY (timestamp, bundle_id, id, other_id, event))")
    cur.execute("CREATE INDEX bundle_events_bundle_id_index ON bundle_events(bundle_id)")

    cur.execute("DROP TABLE IF EXISTS bundles")
    cur.execute("CREATE TABLE bundles (bundle_id TEXT PRIMARY KEY, type TEXT NOT NULL, send TIMESTAMP, src_id INTEGER, on_nodes INTEGER)")

    cur.execute("DROP TABLE IF EXISTS app_data")
    cur.execute("CREATE TABLE app_data (timestamp TIMESTAMP NOT NULL, sent_time TIMESTAMP, id INTEGER NOT NULL, other_id INTEGER, received INTEGER, type TEXT NOT NULL, msg_id TEXT, size INTEGER, latitude REAL, longitude REAL, category TEXT, text1 TEXT, text2 TEXT, text3 TEXT, realm_creator TEXT, status INTEGER, PRIMARY KEY (timestamp, sent_time, id, other_id, type, msg_id))")

    cur.execute("DROP TABLE IF EXISTS sensor_data")
    cur.execute("CREATE TABLE sensor_data (timestamp TIMESTAMP NOT NULL, id INTEGER NOT NULL, lux REAL, pressure REAL, acc_x REAL, acc_y REAL, acc_z REAL, gyro_x REAL, gyro_y REAL, gyro_z REAL)")
    cur.execute("CREATE INDEX sensor_data_timestamp_index ON sensor_data(timestamp)")
    cur.execute("CREATE INDEX sensor_data_id_index ON sensor_data(id)")

    conn.commit()


    option_inserts = [[key, "%s" % value] for key, value in config.__dict__.items() if key[0].isupper()]
    cur.executemany("INSERT INTO options (key,value) VALUES (?,?)", option_inserts)


    ids = util.readIds(os.path.join(data_path, "ids.db"))
    cur.executemany("INSERT INTO ids (id, ip, dtn_id, source) VALUES (?,?,?,?)", [[i["id"], i["ip"], i["dtn_id"], i["source"]] for i in ids])


    results = util.forAllMP(os.path.join(data_path, sub_dir), ".db", fuseDatabasesWorker)

    node_inserts = []
    movement_inserts = []
    smoothed_movement_inserts = []
    sensor_data_inserts = []

    for r in results:
        _id = util.getId(ids, ip=r["info"]["ip"])

        node_inserts.append([_id, r["info"]["ip"], r["info"]["imei"], r["info"]["brand"], r["info"]["model"]])

        for g in r["gps_data"]:
            movement_inserts.append([g["timestamp"], _id, g["latitude"], g["longitude"], g["altitude"], g["accuracy"], g["distance"], g["speed"]])

        for g in r["smoothed_gps_data"]:
            smoothed_movement_inserts.append([g["timestamp"], _id, g["latitude"], g["longitude"], g["distance"], g["speed"]])

        for i in r["sensor_data"]:
            sensor_data_inserts.append([i["timestamp"], _id, i["lux"], i["pressure"], i["acc_x"], i["acc_y"], i["acc_z"], i["gyro_x"], i["gyro_y"], i["gyro_z"]])

    cur.executemany("INSERT INTO nodes (id, ip, imei, brand, model) VALUES (?,?,?,?,?)", node_inserts)
    cur.executemany("INSERT INTO movements (timestamp, id, latitude, longitude, altitude, accuracy, distance, speed) VALUES (?,?,?,?,?,?,?,?)", movement_inserts)
    cur.executemany("INSERT INTO smoothed_movements (timestamp, id, latitude, longitude, distance, speed) VALUES (?,?,?,?,?,?)", smoothed_movement_inserts)
    cur.executemany("INSERT INTO sensor_data (timestamp, id, lux, pressure, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z) VALUES (?,?,?,?,?,?,?,?,?,?)", sensor_data_inserts)

    cur.executemany("INSERT OR IGNORE INTO nodes (id, ip) VALUES (?,?)", [[i["id"], i["ip"]] for i in ids])  # make sure all ids are in nodes table


    n_rename = {"neighbors_gps_data":"neighbors_movements", "neighbors_smoothed_gps_data":"neighbors_smoothed_movements"}
    n_conn, n_cur = util.open_sqlite(os.path.join(data_path, "neighbors.db"))
    for table in n_rename:
        n_cur.execute("PRAGMA table_info(%s)" % table)
        columns = n_cur.fetchall()

        cur.execute("DROP TABLE IF EXISTS %s" % n_rename[table])

        create_sql = "CREATE TABLE %s (" % n_rename[table]
        insert_sql = "INSERT INTO %s (" % n_rename[table]
        values_sql = ""
        for c in columns:
            column_name = c["name"].split("_")[0]
            create_sql += "'%s' %s," % (column_name, c["type"])
            insert_sql += "'%s'," % column_name
            values_sql += "?,"
        create_sql = create_sql[:-1] + ")"
        insert_sql = insert_sql[:-1] + ") VALUES (%s)" % values_sql[:-1]

        cur.execute(create_sql)

        n_cur.execute("SELECT * FROM %s" % table)
        data = n_cur.fetchall()

        cur.executemany(insert_sql, data)
    n_conn.close()


    c_conn, c_cur = util.open_sqlite(os.path.join(data_path, "connections.db"))
    c_cur.execute("SELECT * FROM connections")
    data = c_cur.fetchall()
    c_conn.close()
    cur.executemany("INSERT INTO connections (start_t, end_t, duration, conn_id, id1, id2, up_distance, down_distance, flag) VALUES (?,?,?,?,?,?,?,?,?)", data)


    m_conn, m_cur = util.open_sqlite(os.path.join(data_path, "messages.db"))
    m_cur.execute("SELECT * FROM message_events")
    data = m_cur.fetchall()
    cur.executemany("INSERT INTO bundle_events (timestamp, bundle_id, id, other_id, event, event_reason) VALUES (?,?,?,?,?,?)", data)

    m_cur.execute("SELECT bundle_id, type, send, src_id, on_nodes FROM bundles")
    data = m_cur.fetchall()
    cur.executemany("INSERT INTO bundles (bundle_id, type, send, src_id, on_nodes) VALUES (?,?,?,?,?)", data)

    m_conn.close()


    r_conn, r_cur = util.open_sqlite(os.path.join(data_path, "realm.db"))
    r_cur.execute("SELECT * FROM realm_messages")
    data = r_cur.fetchall()
    r_conn.close()
    cur.executemany("INSERT INTO app_data (timestamp, sent_time, id, other_id, received, type, msg_id, size, latitude, longitude, category, text1, text2, text3, realm_creator, status) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", data)


    conn.commit()
    cur.execute("VACUUM")  # clean up database (reduces file size)
    conn.close()

    logging.info("Created smarter.db in %.2f seconds" % (datetime.now()-start_time).total_seconds())


if __name__ == "__main__":

    # fuse all generated tables into one database for easier sharing, also rename some things
    fuseDatabases(config.DATA_PATH, config.DATA_SUB_DIR, os.path.join(config.DATA_PATH,"smarter.db"))

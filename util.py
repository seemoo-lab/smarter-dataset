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
import multiprocessing
from multiprocessing.pool import ThreadPool
import time
import logging
import sqlite3
import re

import numpy
import pandas

from config import POOL_SIZE


def worker_wrapper(func, args):
    """Wrapper for multiprocessing worker functions, used for terminating with ^C"""
    try:
        return func(*args)
    except KeyboardInterrupt:
        return


def forAll(input_dir, file_ending, worker, args=None, pool_size=POOL_SIZE, use_threads=True):
    """Executes a function for each file in a folder, utilizing a pool of processes or threads"""
    start = time.time()

    logging.debug("Starting %s ..." % worker.__name__)

    results = []
    processes = []

    if use_threads:
        pool = ThreadPool(processes=pool_size)
    else:
        pool = multiprocessing.Pool(processes=pool_size)

    if not os.path.isdir(input_dir):
        raise Exception("Could not find input_dir %s" % input_dir)

    for f in os.listdir(input_dir):
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
                func_args = (absPath,)
            else:
                func_args = (absPath,) + args

            p = pool.apply_async(func=worker_wrapper, args=(worker, func_args))
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

    logging.debug("All %s finished after %.2f seconds" % (worker.__name__, time.time()-start))

    return results


def forAllMP(input_dir, file_ending, target, args=None, pool_size=POOL_SIZE):
    """forAll multiprocess shortcut"""
    return forAll(input_dir, file_ending, target, args, pool_size, use_threads=False)


def forAllT(input_dir, file_ending, target, args=None, pool_size=POOL_SIZE):
    """forAll multithread shortcut"""
    return forAll(input_dir, file_ending, target, args, pool_size, use_threads=True)


def parallelized_apply(dataframes, worker, args=None, pool_size=POOL_SIZE, use_threads=True):
    """Applies a function to a pandas dataframe, by spliting it and working with multiple threads or processes on it"""
    start = time.time()

    if isinstance(dataframes, list):
        df_parts = dataframes
    else:
        df_parts = numpy.array_split(dataframes, pool_size)

    processes = []

    if use_threads:
        pool = ThreadPool(processes=pool_size)
    else:
        pool = multiprocessing.Pool(processes=pool_size)

    for part in df_parts:
        if args is None:
            func_args = (part,)
        else:
            func_args = (part,) + args
        p = pool.apply_async(func=worker_wrapper, args=(worker, func_args))
        processes.append(p)

    df_list = []
    for p in processes:
        try:
            result = p.get()
        except KeyboardInterrupt:
            logging.info("Got ^C terminating...")
            pool.terminate()
            sys.exit(-1)

        if result is not None:
            df_list.append(result)
    df = pandas.concat(df_list)

    pool.close()

    logging.debug("All %s finished after %.2f seconds." % (worker.__name__, time.time()-start))

    return df


def extractIp(s):
    """Extracts a ipv4 address from a string"""
    ip = None

    ip_match = re.search("[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}", s)
    if ip_match:
        ip = ip_match.group(0)
    else:
        logging.error("Could not get IP from '%s'" % s)

    return ip


def readIds(ids_db):
    """Reads the contents of the ids.db and returns it as a list of dicts"""
    # just read in ids.db once
    if not os.path.isfile(ids_db):
        raise Exception("Coud not find ids database at %s" % ids_db)
    conn = sqlite3.connect(ids_db)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("SELECT * FROM ids")
    ids = cur.fetchall()
    ids = [dict(i) for i in ids]  # convert to pickable list of dicts
    conn.close()

    return ids


def getInfos(ids, id=None, ip=None, dtn_id=None):
    """Get all matching ids for a specific id, ip or dtn_id"""
    if id is not None and ip is None and dtn_id is None:
        infos = [i for i in ids if i["id"] == id]
    elif id is None and ip is not None and dtn_id is None:
        infos = [i for i in ids if i["ip"] == ip]
    elif id is None and ip is None and dtn_id is not None:
        infos = [i for i in ids if i["dtn_id"] == dtn_id]
    else:
        raise Exception("Invalid arguments. Choose only one of [id, ip, dtn_id]")
    return infos


def getIp(ids, id=None, dtn_id=None):
    """Get the matching ip for a id or dtn_id"""
    if id is not None and dtn_id is None:
        infos = getInfos(ids, id=id)
    elif id is None and dtn_id is not None:
        infos = getInfos(ids, dtn_id=dtn_id)
    else:
        raise Exception("Invalid arguments. Choose only one of [id, dtn_id]")

    if len(infos) == 0:
        return None

    ips = []
    for i in infos:
        if i["ip"] not in ips:
            ips.append(i["ip"])

    if len(ips) == 1:
        return ips[0]
    else:
        raise Exception("Could not find unique ip for id:%s, dtn_id:%s" % (id, dtn_id))


def getId(ids, ip=None, dtn_id=None):
    """Get the matching id for a ip or dtn_id"""
    if ip is not None and dtn_id is None:
        infos = getInfos(ids, ip=ip)
    elif ip is None and dtn_id is not None:
        infos = getInfos(ids, dtn_id=dtn_id)
    else:
        raise Exception("Invalid arguments. Choose only one of [ip, dtn_id]")

    if len(infos) == 0:
        return None

    ids = []
    for i in infos:
        if i["id"] not in ids:
            ids.append(i["id"])

    if len(ids) == 1:
        return ids[0]
    else:
        raise Exception("Could not find unique id for ip:%s, dtn_id:%s" % (ip, dtn_id))


def getDtnIds(ids, id=None, ip=None):
    """Get all matching dtn_ids for a id or ip"""
    if id is not None and ip is None:
        infos = getInfos(ids, id=id)
    elif id is None and ip is not None:
        infos = getInfos(ids, ip=ip)
    else:
        raise Exception("Invalid arguments. Choose only one of [id, ip]")

    if infos is None:
        return None

    return sorted([i["dtn_id"] for i in infos])


def open_sqlite(file, create=False, max_speed=False):
    """Open a sqlite database with auto date parsing and set some speedup settings if needed"""
    if not create and not os.path.isfile(file):
        raise Exception("Could not find sqlite database at %s" % file)
    conn = sqlite3.connect(file, detect_types=sqlite3.PARSE_DECLTYPES, timeout=60)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    if max_speed:
        conn.isolation_level = None
        cur.execute("PRAGMA journal_mode=WAL")
        cur.execute("PRAGMA synchronous=OFF")
        cur.execute("PRAGMA cache_size=10000")


    return (conn, cur)


def sqlite_to_df(conn, table, select="*", where=None, order=None, index=None, parse_s=None, parse_e=None, warn_empty=True):
    """Creates a pandas dataframe from an sqlite query, with some dateparsing and index setup"""
    if parse_s is not None and not isinstance(parse_s, list):
        raise ValueError("Parameter parse_s has to be a list of columns or None")
    if parse_e is not None and not isinstance(parse_e, list):
        raise ValueError("Parameter parse_e has to be a list of columns or None")

    self_opened = False
    if not isinstance(conn, sqlite3.Connection):
        if os.path.isfile(conn):
            filename = os.path.basename(conn)
            conn = sqlite3.connect(conn)
            self_opened = True
        else:
            raise Exception("Could not find sqlite db at %s" % conn)
    else:
        cur = conn.cursor()
        cur.execute("PRAGMA database_list")
        filename = os.path.basename(cur.fetchone()[2])

    try:
        query = "SELECT %s FROM %s" % (select, table)
        if where:
            query += " WHERE %s" % where
        if order:
            query += " ORDER BY %s" % order

        parse_dates = {}
        date_format = "%Y-%m-%d %H:%M:%S.%f"
        if index == "timestamp":
            parse_dates["timestamp"] = date_format

        if parse_s:
            for column in parse_s:
                parse_dates[column] = date_format

        if parse_e:
            for column in parse_e:
                parse_dates[column] = "s"

        if parse_dates:
            df = pandas.read_sql(query, conn, parse_dates=parse_dates, index_col=index)
        else:
            df = pandas.read_sql(query, conn, index_col=index)
    except pandas.io.sql.DatabaseError as e:
        if "no such table" in str(e):
            logging.warning("%s has no table %s" % (filename, table))
            df = None
        else:
            raise e

    if df is not None and len(df) <= 0:
        if warn_empty:
            logging.warning("%s empty query" % (filename))
        df = None

    if self_opened:
        conn.close()

    return df


def query_sqlite(cur, query, params=[], if_no_table=None):
    """Queries a sqlite database and returns a specific value if the corresponding table does not exist"""
    try:
        cur.execute(query, params)
        return cur.fetchall()
    except sqlite3.OperationalError as e:
        if "no such table" in str(e):
            return if_no_table
        else:
            raise e

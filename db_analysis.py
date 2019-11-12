import os
import sqlite3
import requests
import sqlalchemy
import numpy as np
from PyQt5 import QtCore
from lxml import html
import sqlalchemy.orm as orm
from sqlalchemy import Table, Column, Integer, String, Boolean
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class LotteryDatabase:

    signal_db_error = QtCore.pyqtSignal(int)

    def __init__(self, db):

        if not os.path.exists('database'):
            os.makedirs('database')

        self.conn = sqlite3.connect(db)
        self.c = self.conn.cursor()

    #     self.engine = sqlalchemy.create_engine(str.format('sqlite:///{}', db), echo=True)
    #     self.metadata = sqlalchemy.MetaData(bind=self.engine)
    #     self.session = orm.sessionmaker(bind=self.engine)
    #
    #     self.initial_db()
    #
    # def initial_db(self):
    #
    #     if not self.engine.dialect.has_table(self.engine, 'USER_PROFILES'):
    #
    #         my_session = orm.Session()
    #
    #         Table('USER_PROFILES', self.metadata,
    #               Column('Id', Integer, primary_key=True),
    #               Column('name', String),
    #               Column('active', Boolean))
    #
    #         self.metadata.create_all()
    #
    #         my_session.add(UserProfile(id=1, name='default', active=True))
    #         my_session.commit()

    def db_execute(self, query):

        try:
            self.c.execute(query)
        except Exception as exp:
            self.signal_db_error.emit('Error!', exp)

    def db_fetch_tables(self):

        self.c.execute("SELECT name "
                       "FROM sqlite_master "
                       "WHERE type='table';")

        return self.c.fetchall()

    def db_get_latest(self, query):

        self.c.execute(str.format('SELECT * '
                                  'FROM {} '
                                  'ORDER BY ID '
                                  'DESC LIMIT 1', query))

        return self.c.fetchone()

    def db_get_length(self, query):

        self.c.execute(str.format('SELECT * '
                                  'FROM {} '
                                  'ORDER BY ID '
                                  'DESC LIMIT 1', query))

        return self.c.fetchone()[0]

    def db_fetchone(self, query, query_id):

        self.c.execute(str.format('SELECT * '
                                  'FROM {} '
                                  'WHERE ID=?', query), [query_id])

        return self.c.fetchone()

    def db_fetchall(self, query):

        self.c.execute(str.format('SELECT * '
                                  'FROM {}', query))

        return self.c.fetchall()

    def db_commit(self):

        self.conn.commit()

    def db_update_table(self, table_name, column, new_value, ids):

        try:
            self.c.execute(str.format('UPDATE {} '
                                      'SET {}={} '
                                      'WHERE id={}', table_name, column, new_value, ids))
        except Exception as exp:
            self.signal_db_error.emit('Error!', exp)

    def db_import_file(self, table_name, filename, game_size):

        imported = 0
        rejected = 0

        with open(filename) as f:
            lines = f.readlines()
            for line in lines:

                my_array = np.fromstring(line, dtype=int, sep=',')
                my_array = my_array.tolist()

                try:
                    self.c.execute(str.format('INSERT INTO {} '
                                              'VALUES ({})',
                                              table_name, ",".join(['?' for _ in range(game_size)])), my_array)
                    imported += 1
                except:
                    rejected += 1

            self.conn.commit()

        return imported, rejected

    def db_import_la_jolla(self):

        self.db_delete_table('LA_JOLLA')

        ids = 1
        t = [2, 3, 4]
        imported = 0
        rejected = 0

        table_headers = ",".join(['ID INTEGER PRIMARY KEY'] + ['LA_JOLLA_' + str(n) + ' INTEGER' for n in range(1, 6)])

        self.db_create_table('LA_JOLLA', table_headers)

        for int_t in t:

            page = requests.get(str.format('http://ljcr.dmgordon.org/cover/show_cover.php?v=42&k=5&t={}', int_t))

            tree = html.fromstring(page.content)

            jolla = tree.xpath('//pre/text()')

            for j in jolla:
                split_lines = j.splitlines()
                for line in split_lines:

                    my_array = np.fromstring(line, dtype=int, sep=' ')
                    my_array = my_array.tolist()

                    try:
                        self.c.execute(str.format('INSERT INTO LA_JOLLA '
                                                  'VALUES ({})',
                                                  ",".join(['?' for _ in range(6)])), [ids] + my_array)
                        imported += 1
                        ids += 1

                    except:
                        rejected += 1

            self.conn.commit()

        return imported, rejected

    def db_delete_table(self, query):

        try:
            self.c.execute((str.format('DROP TABLE IF EXISTS {}', query)))
        except Exception as exp:
            self.signal_db_error.emit('Error!', exp)

    def db_delete_view(self, query):

        try:
            self.c.execute((str.format('DROP VIEW IF EXISTS {}', query)))
        except Exception as exp:
            self.signal_db_error.emit('Error!', exp)

    def db_insert(self, table_name, query):

        self.c.execute(str.format('INSERT INTO {} '
                                  'VALUES ({})', table_name,
                                  ",".join(["?" for _ in range(1, len(query)+1)])), query)

    def db_create_table(self, table_name, headers):

        try:
            self.c.execute(str.format(""" CREATE TABLE IF NOT EXISTS {} ({}); """, table_name, headers))
        except Exception as exp:
            self.signal_db_error.emit('Error!', exp)

    def db_create_view(self, view_name, headers, table_name):

        try:
            self.c.execute(str.format('CREATE VIEW {} AS SELECT {} '
                                      'FROM {}', view_name, headers, table_name))
        except Exception as exp:
            self.signal_db_error.emit('Error!', exp)

    def __del__(self):

        self.c.close()
        self.conn.close()


class UserProfile(Base):

    __tablename__ = 'USER_PROFILES'

    id = Column(Integer, primary_key=True)
    name = Column(String(50), unique=True)
    active = Column(Boolean)

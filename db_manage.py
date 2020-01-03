import os

import numpy as np
import requests
import sqlalchemy
from sqlalchemy.orm import sessionmaker, joinedload
from sqlalchemy.sql import text
from sqlalchemy import exists, Table, func
from PyQt5 import QtCore
from lxml import html

from db_models import *
from games_config import CONFIG

DATABASE_ENGINE = 'sqlite:///database/MYOPM.db'


class LotteryDatabase:

    signal_db_error = QtCore.pyqtSignal(int)

    def __init__(self, initial=False):

        if not os.path.exists('database'):
            os.makedirs('database')

        self.engine = sqlalchemy.create_engine(DATABASE_ENGINE, echo=True)
        self.session = sessionmaker(bind=self.engine)

        if initial:
            BASE.metadata.create_all(bind=self.engine)
            self._initial_user_profile()
            self._initial_add_games()

    def __del__(self):
        self.engine.dispose()

    def _initial_add_games(self):

        session = self.session()

        for item in CONFIG:

            item_game = CONFIG[item]['game']['game_name']
            game_exist = session.query(exists().where(LottoGame.game_name == item_game)).scalar()

            if not game_exist:
                session.add(LottoGame(**CONFIG[item]['game']))
                session.flush()

                CONFIG[item]['number_groups']['game'] = item_game

                session.add(NumberGroups(**CONFIG[item]['number_groups']))
                session.flush()

                CONFIG[item]['alphabetic_groups']['game'] = item_game

                session.add(AlphabeticGroups(**CONFIG[item]['alphabetic_groups']))
                session.flush()

                for feature in CONFIG[item]['model_features']:
                    CONFIG[item]['model_features'][feature]['game'] = item_game
                    session.add(ModelFeatures(**CONFIG[item]['model_features'][feature]))

                session.commit()
                session.close()

    def _initial_user_profile(self):

        session = self.session()
        user_exists = session.query(exists().where(UserProfile.user_name == 'default')).scalar()

        if not user_exists:
            session.add(UserProfile(user_name='default',
                                    active=True))
            session.flush()

            session.add(UserSettings(user_parent='default',
                                     line_current_game='poland_mini_lotto',

                                     combo_db='',
                                     combo_test_size='',
                                     combo_predict_ml='',
                                     combo_predict_model='',
                                     combo_scoring='',

                                     check_win_loss=False,
                                     check_add_random=False,
                                     check_latest=False,
                                     check_sampling=False,
                                     check_keras=False,

                                     list_model=''))

        session.commit()
        session.close()

    def db_add_record(self, model_cls, query):

        session = self.session()
        session.add(model_cls(**query))
        session.commit()
        session.close()

    def db_add_user(self, user_name, is_active=False):

        session = self.session()
        user_exists = session.query(exists().where(UserProfile.user_name == user_name)).scalar()

        if not user_exists:
            session.add(UserProfile(user_name=user_name, active=is_active))
            session.commit()
            session.close()

    def db_create_table(self, table_name, args):

        try:
            if not self.engine.dialect.has_table(self.engine, table_name):
                BASE.metadata.reflect(self.engine)
                _ = Table(table_name, BASE.metadata, *args)
                BASE.metadata.create_all(bind=self.engine)
        except Exception as exp:
            self.signal_db_error.emit('Error!', exp)

    def db_create_view(self, view_name, headers, table_name):

        try:
            self.db_execute(str.format('CREATE VIEW {} '
                                       'AS SELECT {} '
                                       'FROM {}', view_name, headers, table_name))
        except Exception as exp:
            self.signal_db_error.emit('Error!', exp)

    def db_delete_record(self, model_cls, query):

        session = self.session()
        model_query = session.query(model_cls).filter_by(**query).first()

        if model_query is not None:
            session.query(model_cls).filter_by(**query).delete()
            session.commit()
            session.close()

    def db_delete_table(self, table_name):

        try:
            if self.engine.dialect.has_table(self.engine, table_name):
                BASE.metadata.reflect(self.engine)
                delete_table = BASE.metadata.tables[table_name]
                delete_table.delete()

        except Exception as exp:
            self.signal_db_error.emit('Error!', exp)

    def db_delete_view(self, query):

        try:
            self.db_execute('DROP VIEW IF EXISTS ' + query)
        except Exception as exp:
            self.signal_db_error.emit('Error!', exp)

    def db_execute(self, query):

        try:
            session = self.session()
            session.execute(text(str.format("""{}""", query)))
            session.flush()
            session.commit()
            session.close()

        except Exception as exp:
            self.signal_db_error.emit('Error!', exp)

    def db_fetchall(self, model_cls, query, reflection=False):
        if not reflection:
            return self.session().query(model_cls).filter_by(**query).all()
        else:
            BASE.metadata.reflect(self.engine)
            fetch_table = BASE.metadata.tables[model_cls]
            fetchone = fetch_table.fe.values(**query)
            return self.session().execute(fetchone)

    def db_fetchone(self, model_cls, query, reflection=False):
        if not reflection:
            return self.session().query(model_cls).filter_by(**query).first()
        else:
            BASE.metadata.reflect(self.engine)
            fetch_table = BASE.metadata.tables[model_cls]
            fetchone = fetch_table.insert().values(**query)
            return self.session().execute(fetchone)

    def db_fetch_tables(self):
        return self.engine.table_names()

    def db_get_latest(self, model_cls):
        return self.session().query(model_cls).order_by(model_cls.id.desc()).first()

    def db_get_length(self, model_cls, reflection=False):
        if not reflection:
            return self.session().query(model_cls).count()
        else:
            BASE.metadata.reflect(self.engine)
            table_lenght = BASE.metadata.tables[model_cls]
            return self.session().query(table_lenght.columns['id']).count()

    def db_import_file(self, table_name, filename, game_size):

        imported = 0
        rejected = 0

        session = self.session()
        BASE.metadata.reflect(self.engine)

        with open(filename) as f:

            lines = f.readlines()

            for line in lines:

                query = {}
                my_array = np.fromstring(line, dtype=int, sep=',').tolist()

                try:

                    input_table = BASE.metadata.tables[table_name]

                    query['id'] = my_array[0]
                    for i in range(1, game_size):
                        query['NR_' + str(i)] = my_array[i]
                        i += 1

                    insert = input_table.insert().values(**query)
                    session.execute(insert)
                    session.flush()
                    imported += 1

                except Exception:
                    rejected += 1
                    raise

        session.commit()
        session.close()

        return imported, rejected

    def db_import_la_jolla(self):

        self.db_delete_table('LA_JOLLA')

        t = [2, 3, 4]
        imported = 0
        rejected = 0

        session = self.session()
        BASE.metadata.reflect(self.engine)

        table_headers = ",".join(['ID INTEGER PRIMARY KEY'] + ['LA_JOLLA_' + str(n) + ' INTEGER' for n in range(1, 6)])

        self.db_create_table('LA_JOLLA', table_headers)

        for int_t in t:

            page = requests.get(str.format('http://ljcr.dmgordon.org/cover/show_cover.php?v=42&k=5&t={}', int_t))

            tree = html.fromstring(page.content)

            jolla = tree.xpath('//pre/text()')

            for j in jolla:
                split_lines = j.splitlines()
                for line in split_lines:

                    query = {}
                    my_array = np.fromstring(line, dtype=int, sep=' ').tolist()

                    try:

                        input_table = BASE.metadata.tables['LA_JOLLA']

                        query['id'] = my_array[0]
                        for i in range(1, 5):
                            query['NR_' + str(i)] = my_array[i]
                            i += 1

                        insert = input_table.insert().values(**query)
                        session.execute(insert)
                        session.flush()
                        imported += 1

                    except Exception:
                        rejected += 1
                        raise

        return imported, rejected

    def db_update(self, model_cls, query, query_keys):

        session = self.session()
        model_query = session.query(model_cls).filter_by(**query).first()

        if model_query is not None:
            for key in query_keys:
                setattr(model_query, key, query_keys[key])
            session.commit()
            session.close()

    def db_update_table(self, table_name, column, new_value, ids):

        try:
            self.db_execute(str.format('UPDATE {} '
                                       'SET {}={} '
                                       'WHERE id={}', table_name, column, new_value, ids))
        except Exception as exp:
            self.signal_db_error.emit('Error!', exp)

import os

import numpy as np
import requests
import sqlalchemy
from sqlalchemy.orm import sessionmaker
from sqlalchemy.sql import text
from sqlalchemy import exists, Table, MetaData
from PyQt5 import QtCore
from lxml import html

from db_models import *
from games_config import CONFIG

DATABASE_ENGINE = 'sqlite:///database/MYOPM.db'
DYNAMIC_CLASS = {}


class LotteryDatabase:

    signal_db_error = QtCore.pyqtSignal(int)

    def __init__(self, initial=False):

        if not os.path.exists('database'):
            os.makedirs('database')

        self.engine = sqlalchemy.create_engine(DATABASE_ENGINE, echo=False)
        self.session = sessionmaker(bind=self.engine)
        self.new_session = None
        BASE.metadata.bind = self.engine

        if initial:
            BASE.metadata.create_all()
            self._initial_user_profile()
            self._initial_add_games()
            self._initial_database()

    def __del__(self):
        self.engine.dispose()

    def _initial_add_games(self):

        self.create_new_session()

        for item in CONFIG:

            item_game = CONFIG[item]['game']['name']
            game_exist = self.new_session.query(exists().where(LottoGame.name == item_game)).scalar()

            if not game_exist:
                self.new_session.add(LottoGame(**CONFIG[item]['game']))
                self.session_flush()

                CONFIG[item]['number_groups']['game'] = item_game

                self.new_session.add(NumberGroups(**CONFIG[item]['number_groups']))
                self.session_flush()

                CONFIG[item]['alphabetic_groups']['game'] = item_game

                self.new_session.add(AlphabeticGroups(**CONFIG[item]['alphabetic_groups']))
                self.session_flush()

                for feature in CONFIG[item]['model_features']:
                    CONFIG[item]['model_features'][feature]['game'] = item_game
                    self.new_session.add(ModelFeatures(**CONFIG[item]['model_features'][feature]))

                self.session_commit()

        self.session_close()

    def _initial_database(self):

        self.create_new_session()
        games = [game.name for game in self.new_session.query(LottoGame.name)]

        curr_profile = self.new_session\
            .query(UserProfile)\
            .filter_by(**{'active': True})\
            .first()

        user_settings = curr_profile.user_settings[0].list_model

        for game in games:
            curr_game = self.fetchone(LottoGame, {'name': game})

            # Create model for input table
            input_params = {'__tablename__': 'INPUT_' + curr_game.input_table,
                            'id': Column('id', Integer, primary_key=True)}

            for i in range(1, curr_game.length + 1):
                input_params['NR_' + str(i)] = Column('NR_' + str(i), Integer)

            self.create_class_model(curr_game.input_model, input_params)

            # Create model for training & prediction table

            for table in curr_game.user_tables:
                if table.database_name.startswith('MODEL_'):
                    training_params = {'__tablename__': table.database_name,
                                       'id': Column('id', Integer, primary_key=True)}

                    for selected_feature in user_settings.split('|'):
                        for feature in curr_game.model_features:
                            if feature.name == selected_feature:
                                for i in range(1, feature.length + 1):
                                    training_params[feature.header + str(i)] = Column(feature.header + str(i), Integer)

                    self.create_class_model(curr_game.training_model, training_params)

                elif table.database_name.startswith('PREDICT_'):
                    predict_params = {'__tablename__': table.database_name,
                                      'id': Column('id', Integer, primary_key=True)}

                    for selected_feature in user_settings.split('|'):
                        for feature in curr_game.model_features:
                            if feature.name == selected_feature:
                                for i in range(1, feature.length + 1):
                                    predict_params[feature.header + str(i)] = Column(feature.header + str(i), Integer)

                    self.create_class_model(curr_game.predict_model, predict_params)

        self.meta_create_all()
        self.session_close()

    def _initial_user_profile(self):

        self.create_new_session()
        user_exists = self.new_session\
            .query(exists().where(UserProfile.user_name == 'default'))\
            .scalar()

        if not user_exists:

            user_profile = UserProfile(user_name='default',
                                       active=True)

            user_profile.user_settings.append(UserSettings(user_parent='default',
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

            self.new_session.add(user_profile)
            self.session_commit()

        self.session_close()

    def add_record(self, model_class, params):
        self.new_session.add(model_class(**params))

    def add_user(self, user_name, is_active=False):
        self.create_new_session()

        user_exists = self.new_session\
            .query(exists().where(UserProfile.user_name == user_name))\
            .scalar()

        if not user_exists:
            self.new_session.add(UserProfile(user_name=user_name,
                                             active=is_active))
            self.session_commit()
        self.session_close()

    @staticmethod
    def check_model_exist_by_table_name(table_name):
        for model in BASE._decl_class_registry.values():
            if hasattr(model, '__tablename__') and model.__tablename__ == table_name:
                return True
        return False

    def session_close(self):
        self.new_session.close()
        self.new_session = None

    @staticmethod
    def create_class_model(class_name, params):
        _ = type(class_name, (BASE,), params)
        DYNAMIC_CLASS[class_name] = _

    def create_new_session(self):
        self.new_session = self.session()

    def create_table(self, class_name, table_name, params):
       pass

    @staticmethod
    def delete_model_by_table_name(table_name):
        for model in BASE._decl_class_registry.values():
            if hasattr(model, '__tablename__') and model.__tablename__ == table_name:
                del model._decl_class_registry[model.__name__]
                BASE.metadata.clear()
                if model.__name__ in DYNAMIC_CLASS:
                    del DYNAMIC_CLASS[model.__name__]
                    break

    def delete_record(self, model_class, params):
        model_query = self.new_session.query(model_class).filter_by(**params).first()
        if model_query is not None:
            self.new_session.query(model_class).filter_by(**params).delete()

    @staticmethod
    def delete_table(table_name):
        try:
            BASE.metadata.reflect()
            delete_table = BASE.metadata.tables.get(table_name)
            if delete_table is not None:
                BASE.metadata.drop_all(tables=[delete_table], checkfirst=True)
        except Exception as exc:
            print(exc)

    def execute(self, params):

        try:
            self.create_new_session()
            self.new_session.execute(text(str.format("""{}""", params)))
            self.session_commit()
            self.session_close()

        except Exception as exp:
            self.signal_db_error.emit('Error!', exp)

    def fetchall(self, model_class, params):
        return self.session().query(model_class).filter_by(**params).all()

    def fetchone(self, model_class, params):
        return self.session().query(model_class).filter_by(**params).first()

    def fetch_tables(self):
        return self.engine.table_names()

    def get_latest(self, model_class):
        return self.session().query(model_class).order_by(model_class.id.desc()).first()

    def get_table_length(self, model_class):
        return self.session().query(model_class).count()

    def import_file(self, model_class, filename, game_size):

        self.create_new_session()

        imported, rejected = 0, 0
        input_table = self.set_model_by_table_name(model_class)

        if input_table is not None:

            with open(filename) as f:

                lines = f.readlines()

                for line in lines:

                    my_array = np.fromstring(line, dtype=int, sep=',').tolist()

                    try:

                        params = {'id': my_array[0]}
                        for i in range(1, game_size):
                            params['NR_' + str(i)] = my_array[i]

                        self.add_record(input_table, params)
                        imported += 1

                    except:
                        rejected += 1

            self.session_commit()
            self.session_close()

        return imported, rejected

    def import_la_jolla(self):

        self.delete_table('LA_JOLLA')

        t = [2, 3, 4]
        imported = 0
        rejected = 0

        session = self.session()
        BASE.metadata.reflect(self.engine)

        table_headers = ",".join(['ID INTEGER PRIMARY KEY'] + ['LA_JOLLA_' + str(n) + ' INTEGER' for n in range(1, 6)])

        for int_t in t:

            page = requests.get(str.format('http://ljcr.dmgordon.org/cover/show_cover.php?v=42&k=5&t={}', int_t))

            tree = html.fromstring(page.content)

            jolla = tree.xpath('//pre/text()')

            for j in jolla:
                split_lines = j.splitlines()
                for line in split_lines:

                    params = {}
                    my_array = np.fromstring(line, dtype=int, sep=' ').tolist()

                    try:

                        input_table = BASE.metadata.tables['LA_JOLLA']

                        params['id'] = my_array[0]
                        for i in range(1, 5):
                            params['NR_' + str(i)] = my_array[i]
                            i += 1

                        insert = input_table.insert().values(**params)
                        session.execute(insert)
                        session.flush()
                        imported += 1

                    except Exception:
                        rejected += 1
                        raise

        return imported, rejected

    def limit_offset_query(self, class_model, limit, offset):
        return self.new_session.query(class_model)\
                                .filter(class_model.id >= offset)\
                                .filter(class_model.id <= limit).all()

    @staticmethod
    def meta_create_all():
        BASE.metadata.reflect()
        BASE.metadata.create_all()

    def session_flush(self):
        self.new_session.flush()

    def session_commit(self):
        self.new_session.commit()

    @staticmethod
    def set_model_by_table_name(table_name):
        for model in BASE._decl_class_registry.values():
            if hasattr(model, '__tablename__') and model.__tablename__ == table_name:
                return model
        return None

    def update_record(self, model_class, params, query_keys):

        session = self.session()
        model_query = session.query(model_class).filter_by(**params).first()

        if model_query is not None:
            for key in query_keys:
                setattr(model_query, key, query_keys[key])
            session.commit()
            session.close()

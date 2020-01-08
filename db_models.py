from sqlalchemy import Column, Integer, String, Boolean, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

BASE = declarative_base()
FRESH_BASE = declarative_base()


class UserProfile(BASE):

    __tablename__ = 'user_profiles'

    user_name = Column('user_name', String(50), primary_key=True)
    active = Column('is_active', Boolean)
    user_settings = relationship('UserSettings', cascade="all,delete")


class UserSettings(BASE):

    __tablename__ = 'user_settings'

    id = Column('id', Integer, primary_key=True)
    user_parent = Column(String, ForeignKey('user_profiles.user_name'))
    line_current_game = Column('line_current_game', String(100))

    combo_db = Column('combo_db', String(100))
    combo_test_size = Column('combo_test_size', String(100))
    combo_predict_model = Column('combo_predict_model', String(100))
    combo_predict_ml = Column('combo_predict_ml', String(100))
    combo_scoring = Column('combo_scoring', String(100))

    check_win_loss = Column('check_win_loss', Integer)
    check_add_random = Column('check_add_random', Integer)
    check_latest = Column('check_latest', Integer)
    check_sampling = Column('check_sampling', Integer)
    check_keras = Column('check_keras', Integer)

    list_model = Column('list_model', String(250))


class LottoGame(BASE):

    __tablename__ = 'lotto_games'

    name = Column('name', String(50), primary_key=True)
    input_table = Column('input_table', String(50), unique=True)
    input_model = Column('input_model', String(50), unique=True)
    training_model = Column('training_model', String(50), unique=True)
    predict_model = Column('predict_model', String(50), unique=True)
    total_numbers = Column('total_numbers', Integer)
    length = Column('length', Integer)

    number_groups = relationship('NumberGroups', cascade="all,delete", lazy='subquery')
    alphabetic_groups = relationship('AlphabeticGroups', cascade="all,delete", lazy='subquery')
    model_features = relationship('ModelFeatures', cascade="all,delete", lazy='subquery')
    user_tables = relationship('DatabaseModels', cascade="all,delete", lazy='subquery')


class NumberGroups(BASE):

    __tablename__ = 'number_groups'

    game = Column(String(100), ForeignKey('lotto_games.name'), primary_key=True)
    first_group = Column('first_group', String(100))
    second_group = Column('second_group', String(100))
    third_group = Column('third_group', String(100))


class AlphabeticGroups(BASE):

    __tablename__ = 'alphabetic_groups'

    game = Column(String(100), ForeignKey('lotto_games.name'), primary_key=True)
    group_A = Column('group_A', String(100))
    group_B = Column('group_B', String(100))
    group_C = Column('group_C', String(100))
    group_D = Column('group_D', String(100))
    group_E = Column('group_E', String(100))
    group_F = Column('group_F', String(100))
    group_G = Column('group_G', String(100))
    group_H = Column('group_H', String(100))
    group_I = Column('group_I', String(100))
    group_J = Column('group_J', String(100))


class ModelFeatures(BASE):

    __tablename__ = 'features'

    id = Column('id', Integer, primary_key=True)
    game = Column(String(100), ForeignKey('lotto_games.name'))
    name = Column('name', String(100))
    length = Column('length', Integer)
    header = Column('header', String(100))


class DatabaseModels(BASE):

    __tablename__ = 'model_database'

    id = Column('id', Integer, primary_key=True)
    game = Column(String(100), ForeignKey('lotto_games.name'))
    database_name = Column('database_name', String(100))

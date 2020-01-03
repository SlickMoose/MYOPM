import sys
import math
import requests
import datetime
from games_config import CONFIG
from db_analysis import LotteryDatabase
import pandas as pd
from dask import dataframe

from sklearn import model_selection

from sklearn.metrics import (accuracy_score,
                             recall_score,
                             precision_score,
                             f1_score,
                             roc_curve,
                             auc, roc_auc_score,
                             confusion_matrix)

from hyperopt import STATUS_OK, Trials, tpe, hp, fmin, space_eval

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import Adadelta, Adam, rmsprop

form = 'form-a4251ea9cfceec0e0003ead731f36e8d'

web_page = 'https://www.lotto.pl/lotto/wyniki-i-wygrane/wyszukaj'

# request_id = str.format('data_losowania%5Bdate%5D={0}&op=&form_build_id={1}&form_id=lotto_wyszukaj_form', date.format("yyyy-mm-dd"), form)

space = {'choice': hp.choice('num_layers', [{'layers': 'two', },
                                            {'layers': 'three',
                                             'units3': hp.choice('units3', range(64, 256)),
                                             'dropout3': hp.uniform('dropout3', .20, .50)}]),

         'units1': hp.choice('units1', range(64, 256)),
         'units2': hp.choice('units2', range(64, 256)),

         'dropout1': hp.uniform('dropout1', .20, .50),
         'dropout2': hp.uniform('dropout2', .20, .50),

         'batch_size': hp.choice('batch_size', range(128, 512)),

         'epochs': 100,
         'optimizer': hp.choice('optimizer', ['adadelta', 'adam', 'rmsprop']),
         'activation': 'relu'
         }


def import_lotto():

    date = datetime.datetime(2019, 3, 2).strftime('%Y-%m-%d')

    payload = {
        'data_losowania[date]': date,
        'id': 'wyszukaj-wyniki-submit'
    }

    response = requests.post(web_page, data=payload)

    with open("requests_results.html", 'w') as f:
        f.write(response.text)


class TestTF:

    def __init__(self):

        self.ldb = LotteryDatabase(CONFIG['database'])

        self.x = None
        self.y = None

        self.x_train = None
        self.x_validation = None
        self.y_train = None
        self.y_validation = None

        self.curr_game = CONFIG['games']['mini_lotto']

        self.table_headers = []
        self.features = self.curr_game['features']

        feat = ['number_map', 'number_cycles', 'cool numbers']

        for i in feat:
            feature_len = self.features[i]['length'] + 1
            feature_header = self.features[i]['header']
            self.table_headers += [feature_header + str(n) + ' INTEGER' for n in range(1, feature_len)]

    def main_tf(self):

        # dataset = dataframe.read_sql_table(table='MODEL_ml', uri='sqlite:///' + config['database'], index_col='ID', npartitions=6)

        dataset = pd.concat(self.generate_df_pieces(self.ldb.conn, 100000, offset=0, ids=5000))
        # dataset.compute()
        array = dataset.values

        self.x = array[:, :len(self.table_headers)]
        self.y = array[:, len(self.table_headers)]

        self.x_train, self.x_validation, self.y_train, self.y_validation = model_selection.train_test_split(
            self.x, self.y, test_size=0.2, random_state=42)

        bayes_trials = Trials()

        best = fmin(fn=self.keras_objective, space=space, algo=tpe.suggest, max_evals=10, trials=bayes_trials)

        print(best)

        for bt in bayes_trials:
            print(bt['result']['loss'])
            print(bt['result']['params'])

    def generate_df_pieces(self, connection, chunk_size, offset, ids):

        last_row = self.ldb.db_get_length('MODEL_ml')
        chunks = int(math.ceil(last_row / chunk_size))
        n_chunk = 1

        self.ldb.db_delete_view('tempView')
        self.ldb.db_create_view('tempView', ",".join(['DRAFT_ID'] + self.table_headers + ['LABEL']), 'MODEL_ml')

        while True:
            print(str.format('Collecting data from database... {} of {}', n_chunk, chunks))
            sql_ct = "SELECT * FROM tempView WHERE DRAFT_ID <= %d limit %d offset %d" % (ids, chunk_size, offset)
            df_piece = pd.read_sql_query(sql_ct, connection)

            if not df_piece.shape[0]:
                break
            yield df_piece

            if df_piece.shape[0] < chunk_size:
                break

            offset += chunk_size
            n_chunk += 1

    def keras_objective(self, params):

        model = Sequential()
        model.add(Dense(output_dim=params['units1'], kernel_initializer='uniform', input_dim=int(self.x_train.shape[1])))
        model.add(Activation(params['activation']))
        model.add(Dropout(params['dropout1']))

        model.add(Dense(output_dim=params['units2'], kernel_initializer="glorot_uniform"))
        model.add(Activation(params['activation']))
        model.add(Dropout(params['dropout2']))

        if params['choice']['layers'] == 'three':
            model.add(Dense(output_dim=params['choice']['units3'], kernel_initializer="glorot_uniform"))
            model.add(Activation(params['activation']))
            model.add(Dropout(params['choice']['dropout3']))

        model.add(Dense(1))
        model.add(Activation('sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer=params['optimizer'])

        model.fit(self.x_train, self.y_train, nb_epoch=params['epochs'], batch_size=params['batch_size'], verbose=2)

        pred_auc = model.predict_proba(self.x_validation, batch_size=params['batch_size'], verbose=2)
        acc = roc_auc_score(self.y_validation, pred_auc)
        print('AUC:', acc)
        sys.stdout.flush()

        return {'loss': -acc, 'params': params, 'status': STATUS_OK}


if __name__ == '__main__':
    TF = TestTF()
    TF.main_tf()



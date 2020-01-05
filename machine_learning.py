import os
import csv
import math
import datetime
from collections import Counter
from heapq import nlargest, nsmallest
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D
import keras.backend.tensorflow_backend as tb
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import Adadelta, Adam, rmsprop
from imblearn.pipeline import make_pipeline
from imblearn.under_sampling import RandomUnderSampler
from hyperopt import STATUS_OK, Trials, tpe, hp, fmin
from sklearn import model_selection, linear_model
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_curve, auc, confusion_matrix

from games_config import CONFIG
from convert import ConvertModel
from db_manage import LotteryDatabase


class MachineLearning:

    def __init__(self, worker):

        self.worker = worker
        self.ldb = LotteryDatabase()

        # variables
        self.x = None
        self.y = None

        self.x_train = None
        self.x_validation = None
        self.y_train = None
        self.y_validation = None

        self.N_FOLDS = 5
        self.MAX_EVALS = 20
        self.RANDOM_STATE = 42
        self.training_size = 15
        self.n_increment = 10
        self.curr_game = CONFIG['games']['mini_lotto']

        # features
        self.table_headers = []
        self.table_name = self.worker.table_name
        self.features = self.curr_game['features']

        for i in range(self.worker.window.list_model.count()):
            feature_len = self.features[self.worker.window.list_model.item(i).text()]['length'] + 1
            feature_header = self.features[self.worker.window.list_model.item(i).text()]['header']
            self.table_headers += [feature_header + str(n) + ' INTEGER' for n in range(1, feature_len)]

    def generate_df_pieces(self, connection, chunk_size, offset, ids):

        last_row = self.ldb.get_table_length(self.table_name)
        chunks = int(math.ceil(last_row / chunk_size))
        n_chunk = 1

        self.ldb.delete_view('tempView')
        self.ldb.create_view('tempView', ",".join(['DRAFT_ID'] + self.table_headers + ['LABEL']), self.table_name)

        while True:
            self.worker.signal_status.emit(str.format('Collecting data from database... {} of {}', n_chunk, chunks))
            sql_ct = "SELECT * FROM tempView WHERE DRAFT_ID <= %d limit %d offset %d" % (ids, chunk_size, offset)
            df_piece = pd.read_sql_query(sql_ct, connection)

            if not df_piece.shape[0]:
                break
            yield df_piece

            if df_piece.shape[0] < chunk_size:
                break

            offset += chunk_size
            n_chunk += 1

        self.worker.signal_status.emit('')

    def embedded_learning(self, input_array, limit=0, draft_id=0):

        original_len = self.ldb.get_table_length('INPUT_' + self.curr_game['database'])

        dataset = pd.concat(self.generate_df_pieces(self.ldb.conn, 100000, 0, original_len-limit))
        array = dataset.values

        self.x = array[:, 1:len(self.table_headers)+1]
        self.y = array[:, len(self.table_headers)+1]

        tb._SYMBOLIC_SCOPE.value = True
        model = self.choose_model(keras=True)

        self.worker.table_name = 'PREDICT_' + self.worker.window.combo_predict_model.currentText()

        convert = ConvertModel(self.worker, list(map(int, input_array.split(" "))), limit)
        convert.create_prediction_model(input_array)

        self.table_name = 'PREDICT_' + self.worker.window.combo_predict_model.currentText()

        output_dataset = pd.concat(self.generate_df_pieces(self.ldb.conn, 10000, 0, 0))
        output_array = output_dataset.values
        output_x = output_array[:, 1:len(self.table_headers)+1]

        original_len = self.ldb.get_table_length(self.worker.table_name) + 1

        now = datetime.datetime.now()
        file_name_r = str.format('{} {}', self.worker.window.combo_predict_model.currentText(),
                               now.strftime("%Y-%m-%d %H %M %S")) + model['info']

        export_columns = ['DRAFT_NR', 'FIRST', 'SECOND', 'THIRD', 'FOURTH', 'FIFTH', 'ODD_EVEN', 'LOW_HIGH', 'LA_JOLLA',
                          'SCORE_ALL', 'SCORE_TOP', 'SCORE_LESS', 'SCORE_2', 'SCORE_3', 'LABEL'] + \
                         ['OUTPUT ' + str(n) for n in range(1, self.training_size+1)]

        with open('archived/' + file_name_r + '.csv', 'a', newline='') as csv_file:

            writer = csv.writer(csv_file)
            writer.writerow(export_columns)

            score_all, score_2, score_3, score_top, score_less = 0, 0, 0, 0, 0

            pairs_two = convert.get_latest_pairs(2)
            pairs_three = convert.get_latest_pairs(3)

            latest_numbers = convert.get_latest_top()
            top_numbers = nlargest(20, latest_numbers, key=latest_numbers.get)
            less_numbers = nsmallest(20, latest_numbers, key=latest_numbers.get)

            head = ','.join(['LA_JOLLA_' + str(n) for n in range(1, 6)])

            # self.ldb.db_delete_view('LA_JOLLA_VIEW')
            # self.ldb.db_create_view('LA_JOLLA_VIEW', head, 'LA JOLLA')
            # self.ldb.db_execute('SELECT * FROM LA_JOLLA_VIEW')

            la_jolla_db = self.ldb.c.fetchall()

            for o in range(1, original_len):
                fetch_one = list(self.ldb.fetchone(self.table_name, o))

                originals = fetch_one[2:self.curr_game['length'] + 2]
                label_column = [fetch_one[-1]]

                output_list = [n + 1 for n in range(0, len(originals)) if originals[n] == 1]

                odd_count = len(list(filter(lambda x: (x % 2 != 0), output_list)))
                even_count = len(list(filter(lambda x: (x % 2 == 0), output_list)))

                if even_count > odd_count:
                    odd_even_check = 1
                else:
                    odd_even_check = 0

                high_low = sum(x > 21 for x in output_list)

                decrease = 0
                for top in top_numbers:
                    if int(top) in output_list:
                        score_all += (1 - decrease)
                        score_top += (1 - decrease)
                    decrease += 0.05

                decrease = 0
                for top in less_numbers:
                    if int(top) in output_list:
                        # score_all += (1 - decrease)
                        score_less += (1 - decrease)
                    decrease += 0.05

                output_counter = Counter(combinations(output_list, 2))

                decrease = 0
                for pair in pairs_two:
                    if pair in output_counter:
                        score_all += (1 - decrease)
                        score_2 += (1 - decrease)
                    decrease += 0.01

                output_counter = Counter(combinations(output_list, 3))

                decrease = 0
                for pair in pairs_three:
                    if pair in output_counter:
                        score_all += (1 - decrease)
                        score_3 += (1 - decrease)
                    decrease += 0.01

                if output_list in la_jolla_db:
                    la_jolla = 1
                else:
                    la_jolla = 0

                output_list = [draft_id] + output_list + [odd_even_check] + [high_low] + [la_jolla] +\
                              [score_all] + [score_top] + [score_less] + [score_2] + [score_3] + label_column

                writer.writerow(output_list)

                score_all, score_2, score_3, score_top, score_less = 0, 0, 0, 0, 0

            self.worker.signal_status.emit('')

        if self.worker.window.check_keras.isChecked():

            model['model'].fit(self.x, self.y, nb_epoch=100, batch_size=212, verbose=2)

            prediction = model['model'].predict(output_x)

            combined_set = list(map(str, prediction))

            with open('archived/' + file_name_r + '.csv', 'r') as read_csv_file:

                csv_input = csv.reader(read_csv_file)
                next(csv_input)

                now = datetime.datetime.now()
                file_name_w = str.format('{} {}', self.worker.window.combo_predict_model.currentText(),
                                         now.strftime("%Y-%m-%d %H %M %S")) + model['info']

                with open('archived/' + file_name_w + '.csv', 'w', newline='') as csv_file:
                    writer = csv.writer(csv_file)
                    writer.writerow(export_columns)

                    for row, o in zip(csv_input, combined_set):
                        writer.writerow(row + [o])

            os.remove('archived/' + file_name_r + '.csv')
            file_name_r = file_name_w

        elif self.worker.window.combo_predict_ml.currentText() == 'LogisticRegression' or \
                self.worker.window.combo_predict_ml.currentText() == 'SGDClassifier':

            model['model'].fit(self.x, self.y)

            prediction = model['model'].predict(output_x)

            combined_set = list(map(str, prediction))

            with open('archived/' + file_name_r + '.csv', 'r') as read_csv_file:

                csv_input = csv.reader(read_csv_file)
                next(csv_input)

                now = datetime.datetime.now()
                file_name_w = str.format('{} {}', self.worker.window.combo_predict_model.currentText(),
                                         now.strftime("%Y-%m-%d %H %M %S")) + model['info']

                with open('archived/' + file_name_w + '.csv', 'w', newline='') as csv_file:
                    writer = csv.writer(csv_file)
                    writer.writerow(export_columns)

                    for row, o in zip(csv_input, combined_set):
                        writer.writerow(row + [o])

            os.remove('archived/' + file_name_r + '.csv')
            file_name_r = file_name_w

        else:

            for t in range(self.training_size):
                self.worker.signal_status.emit(str.format('Training in process... {} of {}', t + 1, self.training_size))
                model['model'].n_estimators += self.n_increment
                model['model'].fit(self.x, self.y)

                prediction = model['model'].predict(output_x)

                combined_set = list(map(str, prediction))

                with open('archived/' + file_name_r + '.csv', 'r') as read_csv_file:

                    csv_input = csv.reader(read_csv_file)
                    next(csv_input)

                    now = datetime.datetime.now()
                    file_name_w = str.format('{} {}', self.worker.window.combo_predict_model.currentText(),
                                             now.strftime("%Y-%m-%d %H %M %S")) + model['info']

                    with open('archived/' + file_name_w + '.csv', 'w', newline='') as csv_file:

                        writer = csv.writer(csv_file)
                        writer.writerow(export_columns)

                        for row, o in zip(csv_input, combined_set):

                            writer.writerow(row + [o])

                os.remove('archived/' + file_name_r + '.csv')
                file_name_r = file_name_w

        self.worker.signal_status.emit('')

        msg = ''

        # msg = "Algorithm: RandomForestClassifier" + '\n' + \
        #       "Number of estimators: {}".format(forest.n_estimators) + '\n' + \
        #       "Accuracy on training set: {:.3f}".format(forest.score(x_train, y_train)) + '\n' + \
        #       "Accuracy on test set: {:.3f}".format(forest.score(x_validation, y_validation))

        return msg

    def choose_model(self, params=None, fresh=False, keras=False):

        model, info = None, None

        if keras:

            model = Sequential()
            model.add(Dense(output_dim=220, kernel_initializer='uniform', input_dim=int(self.x.shape[1])))
            model.add(Activation(activation='relu'))
            model.add(Dropout(0.27208339620963506))

            model.add(Dense(output_dim=205, kernel_initializer="glorot_uniform"))
            model.add(Activation(activation='relu'))
            model.add(Dropout(0.29152160619480066))

            model.add(Dense(1))
            model.add(Activation('sigmoid'))
            model.compile(loss='binary_crossentropy', optimizer='rmsprop')

        elif self.worker.window.list_ml.currentItem().text() == 'RandomForestClassifier':

            if params is not None:
                model = RandomForestClassifier(**params)
            elif fresh:
                model = RandomForestClassifier()
            else:
                model = RandomForestClassifier(warm_start=True, n_estimators=1, n_jobs=-1,
                                               random_state=self.RANDOM_STATE)

            # model = RandomForestClassifier(warm_start=True, n_estimators=1, n_jobs=-1, random_state=self.RANDOM_STATE)
            info = ' RFC ' + self.worker.window.combo_db.currentText()

        elif self.worker.window.list_ml.currentItem().text() == 'RandomForestRegressor':

            if params is not None:
                model = RandomForestRegressor(**params)
            elif fresh:
                model = RandomForestRegressor()
            else:
                model = RandomForestRegressor(warm_start=True, n_estimators=1, n_jobs=-1,
                                              random_state=self.RANDOM_STATE)

            info = ' RFR ' + self.worker.window.combo_db.currentText()

        elif self.worker.window.list_ml.currentItem().text() == 'LogisticRegression':

            if params is not None:
                model = linear_model.LogisticRegression(**params)
            elif fresh:
                model = linear_model.LogisticRegression()
            else:
                model = linear_model.LogisticRegression(C=50, solver='liblinear')
            info = ' LR ' + self.worker.window.combo_db.currentText()

        elif self.worker.window.list_ml.currentItem().text() == 'SGDClassifier':

            if params is not None:
                model = linear_model.SGDClassifier(**params)
            elif fresh:
                model = linear_model.SGDClassifier()
            else:
                model = linear_model.SGDClassifier(class_weight='balanced', loss='hinge', max_iter=2426,
                                               tol=1.6246894453989777e-05, warm_start=True)
            # model = linear_model.SGDClassifier(class_weight='balanced', loss='log', max_iter=2330, tol=7.289319599768096e-05)
            # model = linear_model.SGDClassifier(max_iter=1486, tol=4.663673194605843e-05, loss='log', fit_intercept=False)
            # model = linear_model.SGDClassifier(max_iter=840, tol=15.8197115265907305e-05, class_weight='balanced', loss='modified_huber')
            info = ' SGD ' + self.worker.window.combo_db.currentText()

        return {'model': model, 'info': info}

    def choose_space(self, keras=False):

        space = {}

        if keras:

            space = {'choice': hp.choice('num_layers',
                                         [{'layers': 'two', },
                                          {'layers': 'three',
                                           'units3': hp.uniform('units3', 64, 1024),
                                           'dropout3': hp.uniform('dropout3', .25, .75)}
                                          ]),

                     'units1': hp.uniform('units1', 64, 1024),
                     'units2': hp.uniform('units2', 64, 1024),

                     'dropout1': hp.uniform('dropout1', .25, .75),
                     'dropout2': hp.uniform('dropout2', .25, .75),

                     'batch_size': hp.uniform('batch_size', 28, 128),

                     'nb_epochs': 100,
                     'optimizer': hp.choice('optimizer', ['adadelta', 'adam', 'rmsprop']),
                     'activation': 'relu'
                     }

        elif self.worker.window.list_ml.currentItem().text() == 'RandomForestClassifier':

            space = {
                'n_estimators': hp.choice('n_estimators', range(100, 1500)),
                'class_weight': hp.choice('class_weight', ['balanced', 'balanced_subsample', None]),
                'max_features': hp.choice('max_features', ['auto', 'sqrt', 'log2']),
                'bootstrap': hp.choice('bootstrap', [True, False]),
                'max_depth': hp.choice('max_depth', [None, 1, 3]),
                'criterion': hp.choice('criterion', ['gini', 'entropy'])
            }

        elif self.worker.window.list_ml.currentItem().text() == 'RandomForestRegressor':

            space = {
                'n_estimators': hp.choice('n_estimators', range(10, 150)),
                'warm_start': hp.choice('warm_start', [True, False]),
                'class_weight': hp.choice('class_weight', ['balanced', 'balanced_subsample', None]),
                'max_features': hp.choice('max_features', ['auto', 'sqrt']),
                'bootstrap': hp.choice('bootstrap', [True, False]),
                'max_depth': hp.choice('max_depth', [None, 1, 2, 3]),
                'min_samples_split': hp.choice('min_samples_split', [2, 3]),
                'min_samples_leaf': hp.choice('min_samples_leaf', [1, 2])
            }

        elif self.worker.window.list_ml.currentItem().text() == 'LogisticRegression':

            space = {
                'solver': hp.choice('solver', ['newton-cg', 'lbfgs', 'sag']),
                'warm_start': hp.choice('warm_start', [True, False]),
                'class_weight': hp.choice('class_weight', ['balanced', None]),
                'tol': hp.uniform('tol', 0.00001, 0.0001),
                'C': hp.uniform('C', 1.0, 50.0),
                'fit_intercept': hp.choice('fit_intercept', [True, False]),
                'max_iter': hp.choice('max_iter', range(100, 3000))
            }

        elif self.worker.window.list_ml.currentItem().text() == 'SGDClassifier':

            space = {
                'class_weight': hp.choice('class_weight', [None, 'balanced']),
                'warm_start': hp.choice('warm_start', [True, False]),
                'fit_intercept': hp.choice('fit_intercept', [True, False]),
                'tol': hp.uniform('tol', 0.00001, 0.0001),
                'loss': hp.choice('loss', ['hinge', 'log', 'squared_hinge', 'modified_huber']),
                'max_iter': hp.choice('max_iter', range(500, 3000))
            }

        return space

    def sklearn_model_train(self):

        print(self.worker.window.list_ml.currentItem().text())

        dataset = pd.concat(self.generate_df_pieces(self.ldb.conn, 100000, offset=0, ids=5000))
        array = dataset.values

        self.x = array[:, :len(self.table_headers)]
        self.y = array[:, len(self.table_headers)]

        x_train, x_validation, y_train, y_validation = model_selection.train_test_split(self.x, self.y, test_size=0.2,
                                                                                        random_state=self.RANDOM_STATE)

        space = self.choose_space()

        bayes_trials = Trials()

        best = fmin(fn=self.sklearn_objective, space=space, algo=tpe.suggest, max_evals=self.MAX_EVALS, trials=bayes_trials)

        print(best)

        for bt in bayes_trials:
            print(bt['result']['loss'])
            print(bt['result']['params'])

        model = self.choose_model(space_eval(space, best))
        model['model'].fit(x_train, y_train)
        y_pred = model['model'].predict(x_validation)

        msg = 'Accuracy Score : ' + str(accuracy_score(y_validation, y_pred)) + '\n' + \
              'Precision Score : ' + str(precision_score(y_validation, y_pred)) + '\n' + \
              'Recall Score : ' + str(recall_score(y_validation, y_pred)) + '\n' + \
              'F1 Score : ' + str(f1_score(y_validation, y_pred)) + '\n' + \
              'ROC_AUC Score:' + str(roc_auc_score(y_validation, y_pred))

        # 'Confusion Matrix : \n' + str(confusion_matrix(y_validation, y_pred))

        plt.figure()
        self.plot_confusion_matrix(confusion_matrix(y_validation, y_pred), classes=[1, 0])

        self.worker.signal_status.emit('')

        self.worker.signal_infobox.emit("Completed", msg)
        plt.show()

    def sklearn_objective(self, params, n_folds=5):

        clf = self.choose_model(fresh=True)

        rus = RandomUnderSampler()

        # pipeline = make_pipeline(rus, clf)

        scores = model_selection.cross_val_score(clf['model'], self.x, self.y, cv=n_folds, scoring='f1_macro')

        best_score = max(scores)

        loss = 1 - best_score

        return {'loss': loss, 'params': params, 'status': STATUS_OK}

    def keras_model_train(self):

        dataset = pd.concat(self.generate_df_pieces(self.ldb.conn, 100000, offset=0, ids=5000))
        array = dataset.values

        self.x = array[:, :len(self.table_headers)]
        self.y = array[:, len(self.table_headers)]

        self.x_train, self.x_validation, self.y_train, self.y_validation = model_selection.train_test_split(
            self.x, self.y, test_size=0.2, random_state=self.RANDOM_STATE)

        space = self.choose_space(keras=True)

        tb._SYMBOLIC_SCOPE.value = True

        bayes_trials = Trials()

        best = fmin(fn=self.keras_objective, space=space, algo=tpe.suggest, max_evals=self.MAX_EVALS, trials=bayes_trials)

        print(best)

        for bt in bayes_trials:
            print(bt['result']['loss'])
            print(bt['result']['params'])

    def keras_objective(self, params):

        model = Sequential()
        model.add(Dense(output_dim=params['units1'], input_dim=self.x_train.shape[1]))
        model.add(Activation(params['activation']))
        model.add(Dropout(params['dropout1']))

        model.add(Dense(output_dim=params['units2'], init="glorot_uniform"))
        model.add(Activation(params['activation']))
        model.add(Dropout(params['dropout2']))

        if params['choice']['layers'] == 'three':
            model.add(Dense(output_dim=params['choice']['units3'], init="glorot_uniform"))
            model.add(Activation(params['activation']))
            model.add(Dropout(params['choice']['dropout3']))

        model.add(Dense(1))
        model.add(Activation('sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer=params['optimizer'])

        model.fit(self.x_train, self.y_train, nb_epoch=params['nb_epochs'], batch_size=params['batch_size'], verbose=0)

        pred_auc = model.predict_proba(self.x_validation, batch_size=128, verbose=0)
        acc = roc_auc_score(self.y_validation, pred_auc)
        # print('AUC:', acc)
        # sys.stdout.flush()

        return {'loss': -acc, 'params': params}

    @staticmethod
    def plot_confusion_matrix(cm, classes, normalize=False, cmap=plt.cm.Blues):

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title('Confusion matrix')
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()

    def validate_estimators(self, x, y):

        x_train, x_validation, y_train, y_validation = model_selection.train_test_split(x, y, test_size=0.3,
                                                                                        random_state=0)
        n_estimators = []
        train_results = []
        test_results = []
        rf = RandomForestRegressor(warm_start=True, n_estimators=0, n_jobs=-1)
        # rf = RandomForestClassifier(warm_start=True, n_estimators=0, n_jobs=-1)

        for t in range(self.training_size):

            rf.n_estimators += 3
            # rf.n_iter += 2
            # n_estimators += [rf.n_iter]
            n_estimators += [rf.n_estimators]

            self.worker.signal_status.emit('Validating estimators: {} of {}. Current estimator: {}'.format(
                t + 1, self.training_size, rf.n_estimators))

            rf.fit(x_train, y_train)

            train_pred = rf.predict(x_train)

            false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
            roc_auc = auc(false_positive_rate, true_positive_rate)
            train_results.append(roc_auc)

            y_pred = rf.predict(x_validation)

            false_positive_rate, true_positive_rate, thresholds = roc_curve(y_validation, y_pred)
            roc_auc = auc(false_positive_rate, true_positive_rate)
            test_results.append(roc_auc)

        line1, = plt.plot(n_estimators, train_results, 'b', label="Train AUC")
        line2, = plt.plot(n_estimators, test_results, 'r', label="Test AUC")

        plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
        plt.ylabel('AUC score')
        plt.xlabel('n_estimators')
        plt.show()

    def validate_max_depth(self, x, y):

        x_train, x_validation, y_train, y_validation = model_selection.train_test_split(x, y, test_size=0.3,
                                                                                        random_state=0)

        max_depths = np.linspace(1, 32, 32, endpoint=True)

        train_results = []
        test_results = []

        for max_depth in max_depths:

            rf = RandomForestClassifier(warm_start=True, n_estimators=10, max_depth=max_depth, n_jobs=-1)

            self.worker.signal_status.emit('Validating max depth: {} of {}.'.format(
                max_depth, len(max_depths)))

            rf.fit(x_train, y_train)

            train_pred = rf.predict(x_train)

            false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
            roc_auc = auc(false_positive_rate, true_positive_rate)

            train_results.append(roc_auc)

            y_pred = rf.predict(x_validation)

            false_positive_rate, true_positive_rate, thresholds = roc_curve(y_validation, y_pred)
            roc_auc = auc(false_positive_rate, true_positive_rate)

            test_results.append(roc_auc)

        line1, = plt.plot(max_depths, train_results, 'b', label="Train AUC")
        line2, = plt.plot(max_depths, test_results, 'r', label="Test AUC")

        plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
        plt.ylabel('AUC score')
        plt.xlabel('Tree depth')
        plt.show()

    def validate_min_sample_split(self, x, y):

        x_train, x_validation, y_train, y_validation = model_selection.train_test_split(x, y, test_size=0.3,
                                                                                        random_state=0)

        min_samples_splits = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]

        train_results = []
        test_results = []

        for min_samples_split in min_samples_splits:

            rf = RandomForestClassifier(warm_start=True, n_estimators=10, min_samples_split=min_samples_split, n_jobs=-1)

            self.worker.signal_status.emit('Validating min sample split: {} of {}.'.format(
                min_samples_split, len(min_samples_splits)))

            rf.fit(x_train, y_train)

            train_pred = rf.predict(x_train)
            false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
            roc_auc = auc(false_positive_rate, true_positive_rate)

            train_results.append(roc_auc)

            y_pred = rf.predict(x_validation)

            false_positive_rate, true_positive_rate, thresholds = roc_curve(y_validation, y_pred)
            roc_auc = auc(false_positive_rate, true_positive_rate)
            test_results.append(roc_auc)

        line1, = plt.plot(min_samples_splits, train_results, 'b', label="Train AUC")
        line2, = plt.plot(min_samples_splits, test_results, 'r', label="Test AUC")

        plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
        plt.ylabel('AUC score')
        plt.xlabel('min samples split')
        plt.show()

    def validate_min_sample_leaf(self, x, y):

        x_train, x_validation, y_train, y_validation = model_selection.train_test_split(x, y, test_size=0.3,
                                                                                        random_state=0)

        min_samples_leafs = [1, 2, 3, 4, 5]

        train_results = []
        test_results = []

        for min_samples_leaf in min_samples_leafs:

            rf = RandomForestClassifier(warm_start=True, n_estimators=10, min_samples_leaf=min_samples_leaf, n_jobs=-1)

            self.worker.signal_status.emit('Validating min sample leaf: {} of {}.'.format(
                min_samples_leaf, len(min_samples_leafs)))

            rf.fit(x_train, y_train)
            train_pred = rf.predict(x_train)

            false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)

            roc_auc = auc(false_positive_rate, true_positive_rate)

            train_results.append(roc_auc)
            y_pred = rf.predict(x_validation)

            false_positive_rate, true_positive_rate, thresholds = roc_curve(y_validation, y_pred)

            roc_auc = auc(false_positive_rate, true_positive_rate)

            test_results.append(roc_auc)

        line1, = plt.plot(min_samples_leafs, train_results, 'b', label ="Train AUC")
        line2, = plt.plot(min_samples_leafs, test_results, 'r', label ="Test AUC")

        plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
        plt.ylabel('AUC score')
        plt.xlabel('min samples leaf')
        plt.show()

    def validate_max_features(self):
        pass

    def random_forest_predict(self):

        output_headers = ",".join(['ID INTEGER PRIMARY KEY'] + ['OUTPUT_LABEL INTEGER'])

        self.ldb.delete_table('OUTPUT_prediction')
        self.ldb.create_table('OUTPUT_prediction', output_headers)

        ids = 1

        dataset = pd.concat(self.generate_df_pieces(self.ldb.conn, 1000, offset=0))

        array = dataset.values

        x = array[:, :len(self.table_headers)]

        filename = 'random_forest.sav'
        # loaded_model = joblib.load(filename)
        # prediction = loaded_model.predict(x)

        # combined_set = list(map(str, prediction))

        # for each in combined_set:
        #     self.ldb.db_insert('OUTPUT_prediction', [ids] + [int(each)])
        #     ids += 1
        #
        # self.ldb.db_commit()

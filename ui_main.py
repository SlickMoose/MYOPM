import sys
import inspect
import traceback

import qdarkstyle
from PyQt5 import QtCore, QtGui, uic
from PyQt5.QtWidgets import QApplication, QMessageBox, QFileDialog

from db_models import *
from games_config import *
from convert import ConvertModel
from db_manage import LotteryDatabase
from machine_learning import MachineLearning

Ui_MainWindow, QtBaseClass = uic.loadUiType(r'UI\interface.ui')
Ui_AddDialog, QtBaseAddClass = uic.loadUiType(r'UI\add_feature.ui')
Ui_DbDialog, QtBaseDbClass = uic.loadUiType(r'UI\db_manager.ui')


class Window(QtBaseClass, Ui_MainWindow):

    def __init__(self, parent=None):

        super(QtBaseClass, self).__init__(parent)

        #  class initialize
        self.setupUi(self)
        self.worker = ThreadClass(self)
        self.ldb = LotteryDatabase(True)
        self.update_algorithms()
        self.update_combobox_ml()
        self.get_user_settings()

        # sys.stdout = OutLog(self.stdout_text, sys.stdout, QtGui.QColor(255, 255, 255))
        # sys.stderr = OutLog(self.stdout_text, sys.stderr, QtGui.QColor(255, 255, 255))

        #  variables
        self.select_thread = None
        self.response = None

        # signals
        # self.ldb.signal_db_error.connect(self.info_box)

        self.worker.signal_progress_bar.connect(self.update_progress_bar)
        self.worker.signal_infobox.connect(self.info_box)
        self.worker.signal_status.connect(self.update_status_bar)
        self.worker.signal_qbox.connect(self.question_box)

        # buttons
        self.push_delete.clicked.connect(self.delete_feature)
        self.push_create.clicked.connect(self.create_model)
        self.push_predict.clicked.connect(self.process_input)
        self.push_embedded.clicked.connect(self.process_embedded)
        self.push_add.clicked.connect(self.load_add_ui)
        self.push_ml.clicked.connect(self.sklearn_ml)
        self.push_knn.clicked.connect(self.keras_ml)

        # menu bar actions
        self.actionDatabase_Manager.triggered.connect(self.load_db_manager)
        self.actionExit_Program.triggered.connect(self.close_app)
        self.actionImport_from_file.triggered.connect(self.import_data)
        self.actionVersion.triggered.connect(self.program_version)
        self.actionExport_to.triggered.connect(self.export_to_csv)
        self.actionImport_La_Jolla.triggered.connect(self.update_la_jolla)

        # tooltips
        self.check_add_random.setToolTip('Add random numbers to each sample.')
        self.combo_test_size.setToolTip('Determine testing size for each sample.')

    def save_user_settings(self):

        list_model = '|'.join([str(self.list_model.item(i).text()) for i in range(self.list_model.count())])

        user_config = {'check_win_loss': self.check_win_loss.isChecked(),
                       'check_add_random': self.check_add_random.isChecked(),
                       'check_latest': self.check_latest.isChecked(),
                       'check_sampling': self.check_sampling.isChecked(),
                       'check_keras': self.check_keras.isChecked(),

                       'combo_predict_model': self.combo_predict_model.currentText(),
                       'combo_predict_ml': self.combo_predict_ml.currentText(),
                       'combo_db': self.combo_db.currentText(),
                       'combo_test_size': self.combo_test_size.currentText(),
                       'combo_scoring': self.combo_scoring.currentText(),

                       'list_model': list_model}

        self.ldb.update_record(UserSettings,
                               {'user_parent': 'default',
                            'line_current_game': self.line_current_game.text()},
                               user_config)

    def get_user_settings(self):

        user_config = self.ldb.fetchone(UserSettings, {'user_parent': 'default',
                                                          'line_current_game': self.line_current_game.text()})

        self.check_win_loss.setChecked(user_config.check_win_loss)
        self.check_add_random.setChecked(user_config.check_add_random)
        self.check_latest.setChecked(user_config.check_latest)
        self.check_sampling.setChecked(user_config.check_sampling)
        self.check_keras.setChecked(user_config.check_keras)

        self.combo_predict_model.setCurrentText(user_config.combo_predict_model)
        self.combo_predict_ml.setCurrentText(user_config.combo_predict_ml)
        self.combo_db.setCurrentText(user_config.combo_db)
        self.combo_test_size.setCurrentText(user_config.combo_test_size)
        self.combo_scoring.setCurrentText(user_config.combo_scoring)

        if user_config.list_model != '':
            for each_key in user_config.list_model.split("|"):
                self.list_model.addItem(each_key)

    def load_add_ui(self):
        feature_dialog = ModelAddDialog(self)
        feature_dialog.signal_add_to_model.connect(self.update_add_feature_list)

        for i in range(self.list_model.count()):

            feature_dialog.signal_add_to_list.emit(self.list_model.item(i).text())

        feature_dialog.exec_()

    def load_db_manager(self):
        db_manager = DatabaseManagerDialog(self)
        db_manager.exec_()

    def delete_feature(self):
        self.list_model.takeItem(self.list_model.currentRow())

    def update_add_feature_list(self, item):
        self.list_model.addItem(item)

    def update_algorithms(self):
        self.list_ml.addItem('RandomForestClassifier')
        self.list_ml.addItem('RandomForestRegressor')
        self.list_ml.addItem('LogisticRegression')
        self.list_ml.addItem('SGDClassifier')

    def update_combobox_ml(self):

        tables = self.ldb.fetchall(DatabaseModels, {})

        self.combo_predict_model.clear()
        self.combo_db.clear()
        self.combo_test_size.clear()
        self.combo_predict_ml.clear()

        for alg in range(self.list_ml.count()):
            self.combo_predict_ml.addItem(self.list_ml.item(alg).text())

        for table in tables:
            if table.database_name.startswith('MODEL_'):
                self.combo_db.addItem(table.database_name.replace('MODEL_', ''))
            elif table.database_name.startswith('PREDICT_'):
                self.combo_predict_model.addItem(table.database_name.replace('PREDICT_', ''))

        for n in range(6, 13):
            self.combo_test_size.addItem(str(n))

    def update_status_bar(self, status):
        self.statusBar().showMessage(status)

    def update_progress_bar(self, progress_val):

        if self.select_thread == "process_input":

            self.progress_predict.setValue(progress_val)

        elif self.select_thread == "create_model":

            self.progress_create.setValue(progress_val)

        else:

            self.progress_ml.setValue(progress_val)

    def sklearn_ml(self):
        self.select_thread = inspect.stack()[0][3]
        self.worker.start()

    def keras_ml(self):
        self.select_thread = inspect.stack()[0][3]
        self.worker.start()

    def create_model(self):
        self.select_thread = inspect.stack()[0][3]
        self.worker.start()

    def import_data(self):
        self.select_thread = inspect.stack()[0][3]
        self.worker.start()

    def process_input(self):
        self.select_thread = inspect.stack()[0][3]
        self.worker.start()

    def export_to_csv(self):
        self.select_thread = inspect.stack()[0][3]
        self.worker.start()

    def process_embedded(self):
        self.select_thread = inspect.stack()[0][3]
        self.worker.start()

    def update_la_jolla(self):
        self.select_thread = inspect.stack()[0][3]
        self.worker.start()

    def info_box(self, info_head, info_text):
        QMessageBox.information(self, info_head, info_text)

    def question_box(self, question_head, question_text):

        self.response = QMessageBox.question(
            self, question_head,
            question_text,
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

    def closeEvent(self, event):

        odp = QMessageBox.question(
            self, 'Exit',
            "Are you sure you want exit?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

        if odp == QMessageBox.Yes:
            self.save_user_settings()
            event.accept()
            sys.exit()
        else:
            event.ignore()

    def keyPressEvent(self, event):

        if event.key() == QtCore.Qt.Key_Escape:

            odp = QMessageBox.question(
                self, 'Exit',
                "Are you sure you want exit?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

            if odp == QMessageBox.Yes:
                self.save_user_settings()
                event.accept()
                sys.exit()
            else:
                event.ignore()

    def close_app(self):
        odp = QMessageBox.question(
            self, 'Exit',
            "Are you sure you want exit?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

        if odp == QMessageBox.Yes:
            self.save_user_settings()
            sys.exit()

    def program_version(self):
        self.info_box('Program Version', VERSION)


class ModelAddDialog(QtBaseAddClass, Ui_AddDialog):

    signal_add_to_list = QtCore.pyqtSignal(str)
    signal_add_to_model = QtCore.pyqtSignal(str)

    def __init__(self, window, parent=None):
        super(ModelAddDialog, self).__init__(parent)

        # class initialize
        self.setupUi(self)
        self.window = window
        self.ldb = LotteryDatabase()
        self.list_add_available_init()

        #  signals
        self.signal_add_to_list.connect(self.list_add_selected)

        #  buttons
        self.push_add_ok.clicked.connect(self.add_feature)
        self.push_add_cancel.clicked.connect(self.close_dialog)
        self.feature_add.clicked.connect(self.list_add_selected)
        self.feature_remove.clicked.connect(self.list_remove_selected)

        self.feature_sortUp.clicked.connect(self.move_item_up)
        self.feature_sortDown.clicked.connect(self.move_item_down)

    def list_add_selected(self):
        for item in self.list_add_available.selectedItems():
            if not self.list_feature_order.findItems(item.text(), QtCore.Qt.MatchExactly):
                self.list_feature_order.addItem(item.text())

    def list_remove_selected(self):
        self.list_feature_order.takeItem(self.list_feature_order.currentRow())

    def move_item_up(self):
        if self.list_feature_order.currentRow() > 0:
            current_row = self.list_feature_order.currentRow()
            current_item = self.list_feature_order.takeItem(current_row)
            self.list_feature_order.insertItem(current_row - 1, current_item)
            self.list_feature_order.setCurrentRow(current_row - 1)
            self.list_feature_order.item(current_row - 1).setSelected(True)

    def move_item_down(self):
        if self.list_feature_order.currentRow() < self.list_feature_order.count() - 1:
            current_row = self.list_feature_order.currentRow()
            current_item = self.list_feature_order.takeItem(current_row)
            self.list_feature_order.insertItem(current_row + 1, current_item)
            self.list_feature_order.setCurrentRow(current_row + 1)
            self.list_feature_order.item(current_row + 1).setSelected(True)

    def list_add_available_init(self):
        features = self.ldb.fetchall(ModelFeatures, {'game': self.window.line_current_game.text()})
        for feature in features:
            self.list_add_available.addItem(feature.name)

    def add_feature(self):

        duplicate_check = False
        self.window.list_model.clear()

        for i in range(self.list_feature_order.count()):
            for j in range(self.window.list_model.count()):
                if self.list_feature_order.item(i).text() == self.window.list_model.item(j).text():
                    duplicate_check = True

            if not duplicate_check:
                self.signal_add_to_model.emit(self.list_feature_order.item(i).text())

        self.close_dialog()

    def close_dialog(self):
        self.close()


class DatabaseManagerDialog(QtBaseDbClass, Ui_DbDialog):

    def __init__(self, window, parent=None):
        super(DatabaseManagerDialog, self).__init__(parent)

        # class initialize
        self.setupUi(self)
        self.window = window
        self.ldb = LotteryDatabase()
        self.db_manager_init()

        # variables
        self.deleted = {}
        self.created = {}

        # buttons
        self.btn_add_model.clicked.connect(self.add_model)
        self.btn_delete_model.clicked.connect(self.delete_model)

        self.btn_add_predict.clicked.connect(self.add_predict)
        self.btn_delete_predict.clicked.connect(self.delete_predict)

        self.btn_save.clicked.connect(self.save_database)
        self.btn_cancel.clicked.connect(self.close_db_manager)

    def db_manager_init(self):

        models = self.ldb.fetchall(DatabaseModels, {})

        for model in models:
            if model.database_name.startswith('MODEL_'):
                self.list_model_db.addItem(model.database_name.replace('MODEL_', ''))
            elif model.database_name.startswith('PREDICT_'):
                self.list_predict_db.addItem(model.database_name.replace('PREDICT_', ''))

    def add_model(self):

        text = self.line_model.text().strip()

        if text != '':
            if ' ' in text:
                QMessageBox.information(self,
                                        'Whitespace',
                                        'Your database name contain whitespaces. Please check!')
            else:
                if self.list_model_db.findItems(text, QtCore.Qt.MatchExactly):
                    QMessageBox.information(self,
                                            'Already exist',
                                            'Your database name already exist. Try again!')
                else:
                    self.created[text] = 0
                    self.list_model_db.addItem(text)

    def add_predict(self):

        text = self.line_predict.text().strip()

        if text != '':
            if ' ' in text:
                QMessageBox.information(self,
                                        'Whitespace',
                                        'Your database name contain whitespaces. Please check!')
            else:
                if self.list_predict_db.findItems(text, QtCore.Qt.MatchExactly):
                    QMessageBox.information(self,
                                            'Already exist',
                                            'Your database name already exist. Try again!')
                else:
                    self.created[text] = 1
                    self.list_predict_db.addItem(text)

    def delete_model(self):
        if len(self.list_model_db.selectedItems()) > 0:
            self.deleted[self.list_model_db.currentItem().text()] = 0
            self.list_model_db.takeItem(self.list_model_db.currentRow())

    def delete_predict(self):
        if len(self.list_predict_db.selectedItems()) > 0:
            self.deleted[self.list_predict_db.currentItem().text()] = 1
            self.list_predict_db.takeItem(self.list_predict_db.currentRow())

    def save_database(self):

        self.window.combo_db.clear()
        self.window.combo_predict_model.clear()

        for key, value in self.deleted.items():
            if value == 0:
                self.ldb.delete_record(DatabaseModels, {'database_name': 'MODEL_' + key})
            elif value == 1:
                self.ldb.delete_record(DatabaseModels, {'database_name': 'PREDICT_' + key})

        for key, value in self.created.items():
            if value == 0:
                self.ldb.add_record(DatabaseModels, {'database_name': 'MODEL_' + key})
                self.window.combo_db.addItem(key)
            elif value == 1:
                self.ldb.add_record(DatabaseModels, {'database_name': 'PREDICT_' + key})
                self.window.combo_predict_model.addItem(key)

        self.close_db_manager()

    def close_db_manager(self):
        self.close()


class ThreadClass(QtCore.QThread):

    signal_progress_bar = QtCore.pyqtSignal(int)
    signal_infobox = QtCore.pyqtSignal(str, str)
    signal_status = QtCore.pyqtSignal(str)
    signal_qbox = QtCore.pyqtSignal(str, str)

    def __init__(self, window, parent=None):
        super(ThreadClass, self).__init__(parent)
        self.window = window
        self.table_name = ''

    def run(self):

        process_name = self.window.select_thread

        if process_name == "process_input":

            if self.window.input_line.text() == "":
                self.signal_infobox.emit('Missing input', 'No input numbers to proceed. Please try again.')
            else:
                self.table_name = 'PREDICT_' + self.window.combo_predict_model.currentText()

                try:
                    convert = ConvertModel(self)
                    ml = MachineLearning(self)

                    convert.create_prediction_model(self.window.input_line.text())
                    ml.random_forest_predict()
                    self.signal_progress_bar.emit(0)
                    self.signal_infobox.emit('Completed', 'Prediction model created!')
                except Exception as exc:
                    self.signal_infobox.emit('Error', 'Something went wrong!! ' + str(exc))
                    self.signal_progress_bar.emit(0)

        elif process_name == 'update_la_jolla':

            try:
                ldb = LotteryDatabase()
                imported, rejected = ldb.import_la_jolla()
                self.signal_infobox.emit('Completed', str.format(
                    'Lottery data imported! \n Imported: {} \n Rejected: {}', imported, rejected))

            except Exception as exc:

                self.signal_infobox.emit('Error', 'Something went wrong!! ' + str(exc))
                self.signal_progress_bar.emit(0)

        elif process_name == "create_model":

            if self.window.combo_db.currentText() != '':

                self.table_name = 'MODEL_' + self.window.combo_db.currentText()
                self.signal_qbox.emit('Create', 'Do you want create new model?')

                while self.window.response is None:
                    pass

                if self.window.response == QMessageBox.Yes:
                    self.window.response = None

                    try:
                        convert = ConvertModel(self)

                        if self.window.check_win_loss.isChecked():

                            win, loss = convert.create_training_model()
                            self.signal_progress_bar.emit(0)
                            self.signal_infobox.emit('Completed', 'Training model created! \n' +
                                                     'Win Classification: ' + str(win) + '\n' +
                                                     'Loss Classification: ' + str(loss))

                        else:

                            zero, one, two, three, four = convert.create_training_model()
                            self.signal_progress_bar.emit(0)
                            self.signal_infobox.emit('Completed', 'Training model created! \n' +
                                                     'First Classification: ' + str(zero) + '\n' +
                                                     'Second Classification: ' + str(one) + '\n' +
                                                     'Third Classification: ' + str(two) + '\n' +
                                                     'Fourth Classification: ' + str(three) + '\n' +
                                                     'Fifth Classification: ' + str(four))

                    except Exception as exc:
                        self.signal_infobox.emit('Error', 'Something went wrong!! ' + str(exc))
                        self.signal_progress_bar.emit(0)

        elif process_name == "process_embedded":

            if self.window.input_line.text() == "" and not self.window.check_latest.isChecked():
                self.signal_infobox.emit('Missing input', 'No input numbers to proceed. Please try again.')
            elif self.window.check_latest.isChecked():

                ldb = LotteryDatabase()
                ldb_original = 'INPUT_' + CONFIG['games']['mini_lotto']['database']
                original_len = ldb.get_table_length(ldb_original)

                for i in range(1, 32):

                    try:

                        fetch_one = list(ldb.fetchone(ldb_original, original_len - i + 1))

                        for j in range(self.window.combo_db.count()):

                            self.window.combo_db.setCurrentIndex(j)

                            self.table_name = 'MODEL_' + self.window.combo_db.currentText()
                            self.currentThread().__name__ = "MainThread"
                            ml = MachineLearning(self)
                            _ = ml.embedded_learning(" ".join(map(str, fetch_one[1:6])), i, fetch_one[0])

                    except Exception as exc:
                        self.signal_infobox.emit('Error', 'Something went wrong!! \n' + str(exc))

                self.signal_infobox.emit('Done', 'Finished!!')

            else:

                self.table_name = 'MODEL_' + self.window.combo_db.currentText()
                self.currentThread().__name__ = "MainThread"
                ml = MachineLearning(self)

                try:
                    output = ml.embedded_learning(self.window.input_line.text())
                    self.signal_infobox.emit('Completed', 'Embedded Training finished! \n' + output)
                except Exception as exc:
                    self.signal_infobox.emit('Error', 'Something went wrong!! \n' + str(exc))

        elif process_name == "sklearn_ml":

            if len(self.window.list_ml.selectedItems()) > 0:
                self.table_name = 'MODEL_' + self.window.combo_db.currentText()
                self.currentThread().__name__ = "MainThread"
                ml = MachineLearning(self)

                try:
                    ml.sklearn_model_train()
                    self.signal_infobox.emit('Completed', 'Training model finished!')
                except Exception as exc:
                    self.signal_infobox.emit('Error', 'Something went wrong!! \n' + str(exc))
            else:
                self.signal_infobox.emit('Missing', 'Algorithm has not been selected!')

        elif process_name == "keras_ml":

            self.table_name = 'MODEL_' + self.window.combo_db.currentText()
            self.currentThread().__name__ = "MainThread"
            ml = MachineLearning(self)

            try:
                ml.keras_model_train()
                self.signal_infobox.emit('Completed', 'Training model finished!')
            except Exception as exc:
                self.signal_infobox.emit('Error', 'Something went wrong!! \n' + str(exc))
                print(traceback.format_exc())

        elif process_name == "export_to_csv":

            export_to = ConvertModel(self, False)

            try:
                export_to.convert_to_original()
                self.signal_progress_bar.emit(0)
                self.signal_infobox.emit('Completed', 'Export complete!')
            except Exception as exc:
                self.signal_infobox.emit('Error', 'Something went wrong!! \n' + str(exc))
                self.signal_progress_bar.emit(0)

        elif process_name == "import_data":

            options = QFileDialog.Options()
            options |= QFileDialog.DontUseNativeDialog
            file_name, _ = QFileDialog.getOpenFileName(self.window, "Import file", "",
                                                       "All Files (*);;Text Files (*.txt)", options=options)
            if file_name:

                ldb = LotteryDatabase()
                curr_game = ldb.fetchone(LottoGame, {'name': self.window.line_current_game.text()})

                if not ldb.check_model_exist_by_table_name('INPUT_' + curr_game.input_table):
                    # ldb.delete_table('INPUT_' + curr_game.game_table)
                    # ldb.delete_model(curr_game.input_model)

                    input_params = {'__tablename__': 'INPUT_' + curr_game.input_table,
                                    'id': Column('id', Integer, primary_key=True)}

                    for i in range(1, curr_game.length + 1):
                        input_params['NR_' + str(i)] = Column('NR_' + str(i), Integer)
                    ldb.create_class_model(curr_game.input_model, input_params)
                    ldb.meta_create_all()

                imported, rejected = ldb.import_file(curr_game.input_model, file_name, curr_game.game_len + 1)

                self.signal_infobox.emit('Completed', str.format(
                    'Lottery data imported! \n '
                    'Imported: {} \n '
                    'Rejected: {}',
                    imported, rejected))


class OutLog:

    def __init__(self, edit, out=None, color=None):
        """(edit, out=None, color=None) -> can write stdout, stderr to a
        QTextEdit.
        edit = QTextEdit
        out = alternate stream ( can be the original sys.stdout )
        color = alternate color (i.e. color stderr a different color)
        """
        self.edit = edit
        self.out = out
        self.color = color

    def write(self, m):

        tc = None
        if self.color:
            tc = self.edit.textColor()
            self.edit.setTextColor(self.color)

        # self.edit.moveCursor(QtGui.QTextCursor.End)
        self.edit.insertPlainText(m)

        if self.color:
            self.edit.setTextColor(tc)

        # if self.out:
        #     self.out.write(m)


def main():
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)
    # noinspection PyDeprecation
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    window = Window()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

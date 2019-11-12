import sys
import inspect
import qdarkstyle
from configparser import ConfigParser
from config import config
from convert import ConvertMain
from db_analysis import LotteryDatabase
from machine_learning import MachineLearning
from PyQt5.QtWidgets import QApplication, QMessageBox, QFileDialog, QInputDialog, QLineEdit
from PyQt5 import QtCore, uic

Ui_MainWindow, QtBaseClass = uic.loadUiType(r'UI\interface.ui')
Ui_AddDialog, QtBaseAddClass = uic.loadUiType(r'UI\add_feature.ui')
Ui_DbDialog, QtBaseDbClass = uic.loadUiType(r'UI\db_manager.ui')


class Window(QtBaseClass, Ui_MainWindow):

    def __init__(self, parent=None):

        super(QtBaseClass, self).__init__(parent)

        #  class initialize
        self.setupUi(self)
        self.worker = ThreadClass(self)
        self.ldb = LotteryDatabase(config['database'])
        self.update_algorithms()
        self.update_combobox_ml()
        self.get_user_settings()

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
        self.push_ml.clicked.connect(self.machine_learning)

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
        user_config = ConfigParser()
        user_config['MODEL'] = {'check_add_random': self.check_add_random.isChecked(),
                                'check_win_loss': self.check_win_loss.isChecked(),
                                'combo_db': self.combo_db.currentIndex(),
                                'combo_test_size': self.combo_test_size.currentIndex()
                                }

        user_config['ML ALGORITHMS'] = {}
        for i in range(self.list_ml.count()):
            user_config['ML ALGORITHMS'][self.list_ml.item(i).text()] = self.list_ml.item(i).text()

        user_config['PREDICT'] = {'combo_predict_model': self.combo_predict_model.currentIndex(),
                                  'combo_predict_ml': self.combo_predict_ml.currentIndex()
                                  }

        user_config['FEATURES'] = {}
        for i in range(self.list_model.count()):
            user_config['FEATURES'][self.list_model.item(i).text()] = self.list_model.item(i).text()

        with open(config['user'], 'w') as configfile:
            user_config.write(configfile)

    def get_user_settings(self):

        user_config = ConfigParser()
        user_config.read(config['user'])

        self.check_win_loss.setChecked(user_config.getboolean('MODEL', 'check_win_loss'))
        self.check_add_random.setChecked(user_config.getboolean('MODEL', 'check_add_random'))
        self.combo_predict_model.setCurrentIndex(user_config.getint('PREDICT', 'combo_predict_model'))
        self.combo_predict_ml.setCurrentIndex(user_config.getint('PREDICT', 'combo_predict_ml'))
        self.combo_db.setCurrentIndex(user_config.getint('MODEL', 'combo_db'))
        self.combo_test_size.setCurrentIndex(user_config.getint('MODEL', 'combo_test_size'))

        for each_key in user_config.items('FEATURES'):
            self.list_model.addItem(each_key[1])

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
        # self.list_ml.addItem('RandomForestClassifier')
        # self.list_ml.addItem('RandomForestRegressor')
        # self.list_ml.addItem('LogisticRegression')
        self.list_ml.addItem('SGDClassifier')

    def update_combobox_ml(self):

        tables = self.ldb.db_fetch_tables()
        self.combo_predict_model.clear()
        self.combo_db.clear()
        self.combo_test_size.clear()
        self.combo_predict_ml.clear()

        for alg in range(self.list_ml.count()):
            self.combo_predict_ml.addItem(self.list_ml.item(alg).text())

        for tab in tables:
            if tab[0][:5] == 'MODEL':
                self.combo_db.addItem(tab[0][6:])
            elif tab[0][:7] == 'PREDICT':
                self.combo_predict_model.addItem(tab[0][8:])

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

    def machine_learning(self):
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
        self.info_box('Program Version', config['version'])


class ModelAddDialog(QtBaseAddClass, Ui_AddDialog):

    signal_add_to_list = QtCore.pyqtSignal(str)
    signal_add_to_model = QtCore.pyqtSignal(str)

    def __init__(self, window, parent=None):
        super(ModelAddDialog, self).__init__(parent)

        # class initialize
        self.setupUi(self)
        self.list_add_available_init()
        self.window = window

        #  signals
        self.signal_add_to_list.connect(self.list_add_applied_init)

        #  buttons
        self.push_add_cancel.clicked.connect(self.close_dialog)
        self.push_add_move.clicked.connect(self.move_feature)
        self.push_add_moveback.clicked.connect(self.move_back_feature)
        self.push_add_ok.clicked.connect(self.add_feature)

    def list_add_applied_init(self, item):
        self.list_add_applied.addItem(item)

    def list_add_available_init(self):

        for f in config['games']['mini_lotto']['features']:
            self.list_add_available.addItem(f)

    def add_feature(self):

        duplicate_check = False
        self.window.list_model.clear()

        for i in range(self.list_add_applied.count()):
            for j in range(self.window.list_model.count()):
                if self.list_add_applied.item(i).text() == self.window.list_model.item(j).text():
                    duplicate_check = True

            if not duplicate_check:
                self.signal_add_to_model.emit(self.list_add_applied.item(i).text())

        self.close_dialog()

    def move_feature(self):
        curr_item = self.list_add_available.currentItem().text()

        duplicate_check = False

        for i in range(self.list_add_applied.count()):
            if self.list_add_applied.item(i).text() == curr_item:
                duplicate_check = True

        if not duplicate_check:
            self.list_add_applied.addItem(curr_item)

    def move_back_feature(self):
        self.list_add_applied.takeItem(self.list_add_applied.currentRow())

    def close_dialog(self):
        self.close()


class DatabaseManagerDialog(QtBaseDbClass, Ui_DbDialog):

    def __init__(self, window, parent=None):
        super(DatabaseManagerDialog, self).__init__(parent)

        # class initialize
        self.setupUi(self)
        self.window = window
        self.ldb = LotteryDatabase(config['database'])
        self.db_manager_init()

        # variables
        self.deleted = {}
        self.created = {}

        # buttons
        self.push_db_add.clicked.connect(self.create_database)
        self.push_db_delete.clicked.connect(self.delete_database)
        self.push_db_save.clicked.connect(self.save_database)
        self.push_db_cancel.clicked.connect(self.close_db_manager)

    def db_manager_init(self):

        tables = self.ldb.db_fetch_tables()

        for tab in tables:
            if tab[0][:5] == 'MODEL':
                self.list_model_db.addItem(tab[0][6:])
            elif tab[0][:5] == 'INPUT':
                self.list_input_db.addItem(tab[0][6:])
            else:
                self.list_predict_db.addItem(tab[0][8:])

    def create_database(self):

        text, ok_pressed = QInputDialog.getText(self, "Create Database", "Database Name:", QLineEdit.Normal, "")
        if ok_pressed and text != '':
            text = text.strip()
            if ' ' in text:
                QMessageBox.information(self, 'Whitespace', 'Your database name contain whitespaces. Please check!')
            else:
                if self.db_tabs.currentIndex() == 0:
                    self.created[text] = 0
                    self.list_input_db.addItem(text)
                elif self.db_tabs.currentIndex() == 1:
                    self.created[text] = 1
                    self.list_model_db.addItem(text)
                else:
                    self.created[text] = 2
                    self.list_predict_db.addItem(text)

    def delete_database(self):
        if self.db_tabs.currentIndex() == 0:
            self.deleted[self.list_input_db.currentItem().text()] = 0
            self.list_input_db.takeItem(self.list_input_db.currentRow())
        elif self.db_tabs.currentIndex() == 1:
            self.deleted[self.list_model_db.currentItem().text()] = 1
            self.list_model_db.takeItem(self.list_model_db.currentRow())
        else:
            self.deleted[self.list_model_db.currentItem().text()] = 2
            self.list_model_db.takeItem(self.list_predict_db.currentRow())

    def save_database(self):

        for x, y in self.deleted.items():
            if y == 0:
                self.ldb.db_delete_table(str.format('INPUT_{}', x))
            elif y == 1:
                self.ldb.db_delete_table(str.format('MODEL_{}', x))
            else:
                self.ldb.db_delete_table(str.format('PREDICT_{}', x))

        for x, y in self.created.items():
            if y == 0:
                self.ldb.db_create_table(str.format('INPUT_{}', x), 'ID INTEGER PRIMARY KEY')
            elif y == 1:
                self.ldb.db_create_table(str.format('MODEL_{}', x), 'ID INTEGER PRIMARY KEY')
            else:
                self.ldb.db_create_table(str.format('PREDICT_{}', x), 'ID INTEGER PRIMARY KEY')

        self.window.update_combobox_ml()
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
        if self.window.select_thread == "process_input":

            if self.window.input_line.text() == "":
                self.signal_infobox.emit('Missing input', 'No input numbers to proceed. Please try again.')
            else:
                self.table_name = 'PREDICT_' + self.window.combo_predict_model.currentText()

                try:
                    convert = ConvertMain(self)
                    ml = MachineLearning(self)

                    convert.create_prediction_model(self.window.input_line.text())
                    ml.random_forest_predict()
                    self.signal_progress_bar.emit(0)
                    self.signal_infobox.emit('Completed', 'Prediction model created!')
                except Exception as e:
                    self.signal_infobox.emit('Error', 'Something went wrong!! ' + str(e))
                    self.signal_progress_bar.emit(0)

        elif self.window.select_thread == 'update_la_jolla':

            ldb = LotteryDatabase(config['database'])
            try:

                imported, rejected = ldb.db_import_la_jolla()
                self.signal_infobox.emit('Completed', str.format(
                    'Lottery data imported! \n Imported: {} \n Rejected: {}', imported, rejected))

            except Exception as e:

                self.signal_infobox.emit('Error', 'Something went wrong!! ' + str(e))
                self.signal_progress_bar.emit(0)

        elif self.window.select_thread == "create_model":

            self.table_name = 'MODEL_' + self.window.combo_db.currentText()

            self.signal_qbox.emit('Create', 'Do you want create new model?')

            while self.window.response == '':
                pass

            if self.window.response == QMessageBox.Yes:
                self.window.response = ''

                try:
                    convert = ConvertMain(self)

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

                except Exception as e:
                    self.signal_infobox.emit('Error', 'Something went wrong!! ' + str(e))
                    self.signal_progress_bar.emit(0)

            else:
                pass

        elif self.window.select_thread == "process_embedded":

            if self.window.input_line.text() == "" and not self.window.check_last_20.isChecked():
                self.signal_infobox.emit('Missing input', 'No input numbers to proceed. Please try again.')
            elif self.window.check_last_20.isChecked():

                ldb = LotteryDatabase(config['database'])
                ldb_original = 'INPUT_' + config['games']['mini_lotto']['database']
                original_len = ldb.db_get_length(ldb_original)

                for i in range(1, 32):

                    try:

                        fetch_one = list(ldb.db_fetchone(ldb_original, original_len - i+1))

                        for j in range(self.window.combo_db.count()):

                            self.window.combo_db.setCurrentIndex(j)

                            for m in range(self.window.combo_predict_ml.count()):

                                self.window.combo_predict_ml.setCurrentIndex(m)

                                self.table_name = 'MODEL_' + self.window.combo_db.currentText()
                                self.currentThread().__name__ = "MainThread"
                                ml = MachineLearning(self)
                                output = ml.embedded_learning(" ".join(map(str, fetch_one[1:6])), i, fetch_one[0])

                    except Exception as e:
                        self.signal_infobox.emit('Error', 'Something went wrong!! \n' + str(e))

                self.signal_infobox.emit('Done', 'Finished!!')

            else:

                self.table_name = 'MODEL_' + self.window.combo_db.currentText()
                self.currentThread().__name__ = "MainThread"
                ml = MachineLearning(self)

                try:
                    output = ml.embedded_learning(self.window.input_line.text())
                    self.signal_infobox.emit('Completed', 'Embedded Training finished! \n' + output)
                except Exception as e:
                    self.signal_infobox.emit('Error', 'Something went wrong!! \n' + str(e))

        elif self.window.select_thread == "machine_learning":

            self.table_name = 'MODEL_' + self.window.combo_db.currentText()
            self.currentThread().__name__ = "MainThread"
            ml = MachineLearning(self)

            try:
                ml.random_forest_train()
                self.signal_infobox.emit('Completed', 'Training model finished!')
            except Exception as e:
                self.signal_infobox.emit('Error', 'Something went wrong!! \n' + str(e))

        elif self.window.select_thread == "export_to_csv":

            self.table_name = 'EMPTY'
            export_to = ConvertMain(self)

            try:
                export_to.convert_to_original()
                self.signal_progress_bar.emit(0)
                self.signal_infobox.emit('Completed', 'Export complete!')
            except Exception as e:
                self.signal_infobox.emit('Error', 'Something went wrong!! \n' + str(e))
                self.signal_progress_bar.emit(0)

        elif self.window.select_thread == "import_data":

            options = QFileDialog.Options()
            options |= QFileDialog.DontUseNativeDialog
            file_name, _ = QFileDialog.getOpenFileName(self.window, "Import file", "",
                                                       "All Files (*);;Text Files (*.txt)", options=options)
            if file_name:
                curr_game = config['games']['mini_lotto']['features']['original_numbers']
                thread_ldb = LotteryDatabase(config['database'])
                table_name = 'INPUT_' + config['games']['mini_lotto']['database']
                table = ",".join(['ID INTEGER PRIMARY KEY'] +
                                 ['NR' + str(n) + ' INTEGER' for n in range(1, curr_game['length'] + 1)])

                thread_ldb.db_create_table(table_name, table)
                imported, rejected = thread_ldb.db_import_file(table_name, file_name, curr_game['length'] + 1)

                self.signal_infobox.emit('Completed', str.format(
                    'Lottery data imported! \n Imported: {} \n Rejected: {}', imported, rejected))


def main():
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    window = Window()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

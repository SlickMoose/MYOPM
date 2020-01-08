import csv
import random
import time
import datetime
import math

from PyQt5.QtCore import Qt
from heapq import nlargest, nsmallest
from itertools import combinations
from collections import Counter, defaultdict

from db_models import *
from db_manage import LotteryDatabase, DYNAMIC_CLASS


class ConvertModel:

    def __init__(self, worker, initial_table=True, last_draw=None, limit=0):

        # class initialize
        self.worker = worker
        self.table_name = self.worker.table_name
        self.game_name = self.worker.window.line_current_game.text()

        if initial_table:

            self.ldb = self.worker.window.ldb

            self.curr_game = self.ldb.fetchone(LottoGame, {'name': self.game_name})
            self.input_table = 'INPUT_' + self.curr_game.input_table

            # features
            table_headers = {'__tablename__': self.table_name,
                             'id': Column('id', Integer, primary_key=True),
                             'DRAFT_ID': Column('DRAFT_ID', Integer)}

            for feature in self.curr_game.model_features:
                match_items = self.worker.window.list_model.findItems(feature.name, Qt.MatchExactly)
                if len(match_items) > 0:
                    feature_len = feature.length + 1
                    feature_header = feature.header
                    for n in range(1, feature_len):
                        table_headers[feature_header + str(n)] = Column(feature_header + str(n), Integer)

            table_headers['LABEL'] = Column('LABEL', Integer)

            self.ldb.delete_table(self.table_name)
            self.ldb.delete_model_by_table_name(self.table_name)
            self.ldb.create_class_model(self.curr_game.training_model, table_headers)
            self.ldb.meta_create_all()
            self.input_record_count = self.ldb.get_table_length(DYNAMIC_CLASS[self.curr_game.input_model]) + 1 - limit

            # variables
            if last_draw is None:
                self.last_draw = [13, 18, 29, 32, 37]
            else:
                self.last_draw = last_draw

            self.total_game_numbers = self.curr_game.total_numbers + 1
            self.training_size_limit = int(self.worker.window.combo_test_size.currentText())
            self.single_draw_len = self.curr_game.length + 1

            self.labels = [0, 12, 34, 56]
            self.win = {'LABEL': 1}
            self.loss = {'LABEL': 0}

            self.rash_one = 5
            self.rash_two = 5
            self.rash_three = 5
            self.rash_default = 5

    @staticmethod
    def __append_drawn(current_array, set_of_six):
        new_drawn = []
        for drawn in set_of_six:
            index = drawn - 1
            new_drawn.append(current_array[index])

        return new_drawn

    def __append_rash_group(self, sample):
        new_set = []
        for num in sample:
            if num in self.curr_game['groups']['first_group']:
                new_set += [1]
            elif num in self.curr_game['groups']['second_group']:
                new_set += [2]
            elif num in self.curr_game['groups']['third_group']:
                new_set += [3]

        return new_set

    def __append_alpha_group(self, sample):
        alpha_group = []
        for n in sample:
            for g in self.curr_game['alphabetic_groups'].items():
                if n in g[1]:
                    alpha_group += [1]
                else:
                    alpha_group += [0]

        return alpha_group

    def __map_last_draw(self, curr_draw):
        return {'IN_LAST_' + str(n): i for n, i in zip(range(1, self.single_draw_len), curr_draw)}

    def __append_in_last_draw(self, sample, curr_draw):
        if curr_draw.count(0) == 0:
            last_draw = [0 for _ in range(1, self.total_game_numbers)]
        else:
            last_draw = [1 if n in sample else 0 for n in range(1, self.total_game_numbers)]

        return last_draw

    def __append_numbers_cycle(self, sample, curr_cycle):
        if sum(map((0).__eq__, curr_cycle.values())) == 0:
            numbers_cycle = {'CYCLE_' + str(n): 0 for n in range(1, self.total_game_numbers)}
        else:
            numbers_cycle = {'CYCLE_' + str(n): 1 if n in sample else curr_cycle['CYCLE_' + str(n)]
                             for n in range(1, self.total_game_numbers)}
        return numbers_cycle

    def append_hot_cold_warm_cool(self, top_numbers, n_top, large_small):
        if large_small == 'L':
            hw = nlargest(n_top, top_numbers, key=top_numbers.get)
            hc_wc = [1 if str(n) in hw else 0 for n in range(1, self.total_game_numbers)]
        else:
            cc = nsmallest(n_top, top_numbers, key=top_numbers.get)
            hc_wc = [1 if str(n) in cc else 0 for n in range(1, self.total_game_numbers)]

        return hc_wc

    @staticmethod
    def __append_count_label(sample, count_list):
        label = 0
        count = Counter(sample)
        for x in count_list:
            label = label + count[x]
        return label

    def __map_number_map(self, sample):
        return {'MAP_' + str(n): 1 if n in sample else 0 for n in range(1, self.total_game_numbers)}

    def __map_original_numbers(self, sample):
        return {'ORIGINAL_' + str(n): i for n, i in zip(range(1, self.single_draw_len), sample)}

    def __append_rash(self, array):
        rash_group = []
        for num in array:
            if num in self.curr_game['groups']['first_group']:
                data = 1  # first_data_group.get(num)
                if data is None:
                    per = round(float(1 / self.rash_one) * 100, 2)
                else:
                    per = round(float(data / self.rash_one) * 100, 2)
                rash_group.append(per)

            elif num in self.curr_game['groups']['second_group']:
                data = 1  # second_data_group.get(num)
                if data is None:
                    per = round(float(1 / self.rash_two) * 100, 2)
                else:
                    per = round(float(data / self.rash_two) * 100, 2)
                rash_group.append(per)

            elif num in self.curr_game['groups']['third_group']:
                data = 1  # third_data_group.get(num)
                if data is None:
                    per = round(float(1 / self.rash_three) * 100, 2)
                else:
                    per = round(float(data / self.rash_three) * 100, 2)
                rash_group.append(per)

        return rash_group

    def __append_db(self, params):
        self.ldb.add_record(DYNAMIC_CLASS[self.curr_game.training_model], params)

    def create_prediction_model(self, input_array):

        self.ldb.create_new_session()
        list_model = self.worker.window.list_model

        my_list = list(map(int, input_array.split(" ")))
        my_list = self.__add_random(my_list)

        ids = 1
        combined_set = []
        top_numbers = self.__create_top_numbers(self.input_record_count - 100)
        number_cycles = self.__get_latest_number_cycle()
        curr_draw = self.__get_latest_draw()
        total_combinations = int(math.factorial(42)/(math.factorial(5)*(math.factorial(42-5))))

        for a in my_list:
            # if 1 <= a <= 11:
            for b in my_list:
                if a < b:  # and 5 <= b <= 23:
                    for c in my_list:
                        if b < c:  # and 11 <= c <= 33:
                            for d in my_list:
                                if c < d:  # and 20 <= d <= 40:
                                    for e in my_list:
                                        # diff = e-a
                                        # sample_sum = a+b+c+d+e
                                        if d < e:  # and 29 <= e <= 42: # and 17 < diff < 40 and 69 < sample_sum < 151:
                                            # for f in my_list:
                                            #     if e < f:

                                            sample_array = [a, b, c, d, e]

                                            for i in range(self.worker.window.list_model.count()):

                                                if list_model.item(i).text() == 'number_map':
                                                    combined_set += self.__map_number_map(sample_array)

                                                elif list_model.item(i).text() == 'number_cycles':
                                                    combined_set += self.__append_numbers_cycle(sample_array,
                                                                                                number_cycles)

                                                elif list_model.item(i).text() == 'original_numbers':
                                                    combined_set += sample_array

                                                elif list_model.item(i).text() == 'in_last_draw':
                                                    combined_set += curr_draw

                                                elif list_model.item(i).text() == 'rash_group':
                                                    combined_set += self.__append_rash_group(sample_array)

                                                elif list_model.item(i).text() == 'alphabetic_group':
                                                    combined_set += self.__append_alpha_group(sample_array)

                                                elif list_model.item(i).text() == 'hot numbers':
                                                    combined_set += self.append_hot_cold_warm_cool(
                                                        top_numbers, 10, 'L')

                                                elif list_model.item(i).text() == 'cold numbers':
                                                    combined_set += self.append_hot_cold_warm_cool(
                                                        top_numbers, 10, 'S')

                                                elif list_model.item(i).text() == 'warm numbers':
                                                    combined_set += self.append_hot_cold_warm_cool(
                                                        top_numbers, 20, 'L')

                                                elif list_model.item(i).text() == 'cool numbers':
                                                    combined_set += self.append_hot_cold_warm_cool(
                                                        top_numbers, 20, 'S')

                                            label = [self.__append_count_label(sample_array,
                                                                               self.last_draw)]

                                            self.__append_db(ids, [0], combined_set, label)

                                            combined_set = []

                                            self.worker.signal_progress_bar.emit((ids/total_combinations)*100)

                                            ids += 1

        self.worker.signal_progress_bar.emit(0)
        self.ldb.db_commit()

    def convert_to_original(self):

        self.ldb = LotteryDatabase()
        combo_predict = self.worker.window.combo_predict_model
        self.table_name = 'PREDICT_' + combo_predict.currentText()

        now = datetime.datetime.now()
        file_name = str.format('{} {}', combo_predict.currentText(), now.strftime("%Y-%m-%d %H %M %S"))
        export_columns = ['FIRST', 'SECOND', 'THIRD', 'FOURTH', 'FIFTH', 'SIXTH', 'LABEL', 'OUTPUT']

        with open('archived/' + file_name + '.csv', 'a', newline='') as csv_file:

            writer = csv.writer(csv_file)
            writer.writerow(export_columns)

            for o in range(1, self.input_record_count):
                fetch_one = list(self.ldb.fetchone(self.table_name, o))
                fetch_output = list(self.ldb.fetchone('OUTPUT_prediction', o))

                originals = fetch_one[1:50]
                label_column = [fetch_one[-1]]
                output_column = [fetch_output[-1]]

                output_list = [n + 1 for n in range(0, len(originals)) if originals[n] == 1]
                output_list = output_list + label_column + output_column

                writer.writerow(output_list)

                self.worker.signal_status.emit('Export in progress: {} of {}.'.format(o, self.input_record_count - 1))

        self.worker.signal_status.emit('')

    def __get_latest_number_cycle(self):

        curr_cycle, fetch_one = [], []
        for o in range(1, self.input_record_count):
            curr_cycle = self.__append_numbers_cycle(fetch_one[1:self.single_draw_len], curr_cycle)

            self.ldb.db_commit()
            fetch_one = list(self.ldb.fetchone(self.input_table, o))

        curr_cycle = self.__append_numbers_cycle(fetch_one[1:self.single_draw_len], curr_cycle)
        return curr_cycle

    def __get_latest_draw(self):

        curr_draw, fetch_one = [], []
        for o in range(1, self.input_record_count):
            curr_draw = self.__append_in_last_draw(fetch_one[1:self.single_draw_len], curr_draw)

            self.ldb.session_commit()
            fetch_one = list(self.ldb.fetchone(self.input_table, o))

        curr_draw = self.__append_in_last_draw(fetch_one[1:self.single_draw_len], curr_draw)
        return curr_draw

    def __create_top_numbers(self, offset):

        top_numbers = {}
        last = self.ldb.limit_offset_query(DYNAMIC_CLASS[self.curr_game.input_model], offset, offset-200)
        for sample in last:
            for number in range(1, self.single_draw_len):
                number = str(getattr(sample, 'NR_' + str(number)))
                if number not in top_numbers:
                    top_numbers[number] = 0
                top_numbers[number] += 1

        return top_numbers

    def get_latest_pairs(self, pair_size):

        pairs = {}
        sql_ct = str.format("SELECT * FROM {} limit {} offset {}", self.input_table, 366, self.input_record_count - 367)
        self.ldb.execute(sql_ct)
        last = self.ldb.c.fetchmany(self.input_record_count)
        for sample in last:
            comb = combinations(sample, pair_size)
            for c in comb:
                if c not in pairs:
                    pairs[c] = 1
                else:
                    pairs[c] += 1

        pairs_largest = nlargest(100, pairs, key=pairs.get)
        return pairs_largest

    def get_latest_top(self):
        return self.__create_top_numbers(self.input_record_count - 100)

    def __add_random(self, o_num, limit=True):

        if limit:
            sample_size = self.training_size_limit
        else:
            sample_size = self.total_game_numbers + 1

        while True:
            r = random.randrange(1, self.total_game_numbers)
            if r not in o_num:
                o_num = o_num + [r]
                if len(o_num) == sample_size:
                    o_num.sort()
                    break
        return o_num

    def create_training_model(self):

        self.ldb.create_new_session()
        list_model = self.worker.window.list_model

        ids = 1
        avg_time = 0
        win_count, loss_count = 0, 0
        zero, one, two, three, four = 0, 0, 0, 0, 0

        combined_set, curr_cycle = {}, {}
        fetch_one, curr_draw = [], []
        start_time = time.time()
        
        for o in range(1, self.input_record_count):

            curr_cycle = self.__append_numbers_cycle(fetch_one, curr_cycle)
            curr_draw = self.__append_in_last_draw(fetch_one, curr_draw)
            top_numbers = self.__create_top_numbers(o)

            record_set = self.ldb.fetchone(DYNAMIC_CLASS[self.curr_game.input_model], {'id': o})
            fetch_one = [getattr(record_set, 'NR_' + str(i)) for i in range(1, self.single_draw_len)]

            my_list = self.__add_random(fetch_one)

            end_time = time.time()
            avg_time = (avg_time + (end_time - start_time)) / o
            eta = avg_time * self.input_record_count - avg_time * o

            self.worker.signal_status.emit(self.__print_run_time(eta))
            self.worker.signal_progress_bar.emit(((o + 1) / self.input_record_count) * 100)

            for a in my_list:
                for b in my_list:
                    if a < b:
                        for c in my_list:
                            if b < c:
                                for d in my_list:
                                    if c < d:
                                        for e in my_list:
                                            if d < e:

                                                sample_array = [a, b, c, d, e]

                                                for i in range(self.worker.window.list_model.count()):

                                                    if list_model.item(i).text() == 'number_map':
                                                        combined_set = {**combined_set,
                                                                        **self.__map_number_map(sample_array)}

                                                    elif list_model.item(i).text() == 'number_cycles':
                                                        combined_set = {**combined_set,
                                                                        **self.__append_numbers_cycle(
                                                                            fetch_one, curr_cycle)}

                                                    elif list_model.item(i).text() == 'original_numbers':
                                                        combined_set = {**combined_set,
                                                                        **self.__map_original_numbers(sample_array)}

                                                    elif list_model.item(i).text() == 'in_last_draw':
                                                        combined_set = {**combined_set,
                                                                        **self.__map_last_draw(curr_draw)}

                                                    elif list_model.item(i).text() == 'rash_group':
                                                        combined_set = {**combined_set,
                                                                        **self.__append_rash_group(sample_array)}

                                                    elif list_model.item(i).text() == 'alphabetic_group':
                                                        combined_set = {**combined_set,
                                                                        **self.__append_alpha_group(sample_array)}

                                                    elif list_model.item(i).text() == 'hot numbers':
                                                        combined_set = {**combined_set,
                                                                        **self.append_hot_cold_warm_cool(
                                                                            top_numbers, 10, 'L')}

                                                    elif list_model.item(i).text() == 'cold numbers':
                                                        combined_set = {**combined_set,
                                                                        **self.append_hot_cold_warm_cool(
                                                                            top_numbers, 10, 'S')}

                                                    elif list_model.item(i).text() == 'warm numbers':
                                                        combined_set = {**combined_set,
                                                                        **self.append_hot_cold_warm_cool(
                                                                            top_numbers, 20, 'L')}

                                                    elif list_model.item(i).text() == 'cool numbers':
                                                        combined_set = {**combined_set,
                                                                        **self.append_hot_cold_warm_cool(
                                                                            top_numbers, 20, 'S')}

                                                label = [self.__append_count_label(sample_array, fetch_one)]

                                                if self.worker.window.check_win_loss.isChecked():

                                                    if label < [3]:
                                                        self.__append_db({**{'DRAFT_ID': o},
                                                                          **combined_set,
                                                                          **self.loss})
                                                        loss_count += 1
                                                    else:
                                                        self.__append_db({**{'DRAFT_ID': o},
                                                                          **combined_set,
                                                                          **self.win})
                                                        win_count += 1

                                                    ids += 1

                                                else:

                                                    if label[0] in [0, 1]:  # and number_limit[0] < 25:
                                                        self.__append_db({**{'DRAFT_ID': o},
                                                                          **combined_set,
                                                                          **{'LABEL': 0}})
                                                        # number_limit[0] = number_limit[0] + 1
                                                        zero += 1
                                                        ids += 1
                                                    elif label[0] == 2:  # and number_limit[1] < 25:
                                                        self.__append_db({**{'DRAFT_ID': o},
                                                                          **combined_set,
                                                                          **{'LABEL': 1}})
                                                        # number_limit[1] = number_limit[1] + 1
                                                        one += 1
                                                        ids += 1
                                                    elif label[0] == 3:  # and number_limit[1] < 25:
                                                        self.__append_db({**{'DRAFT_ID': o},
                                                                          **combined_set,
                                                                          **{'LABEL': 2}})
                                                        # number_limit[1] = number_limit[1] + 1
                                                        two += 1
                                                        ids += 1
                                                    elif label[0] in [4, 5]:  # and number_limit[2] < 25:
                                                        self.__append_db({**{'DRAFT_ID': o},
                                                                          **combined_set,
                                                                          **{'LABEL': 3}})
                                                        # number_limit[2] = number_limit[2] + 1
                                                        three += 1
                                                        ids += 1

                                                combined_set = {}

        self.ldb.session_commit()
        self.ldb.session_close()
        if self.worker.window.check_win_loss.isChecked():
            return win_count, loss_count
        else:
            return zero, one, two, three, four

    @staticmethod
    def __print_run_time(seconds):
        seconds = int(seconds)
        hours = seconds // 3600
        minutes = (seconds - 3600 * hours) // 60
        seconds = seconds - 3600 * hours - 60 * minutes
        print_it = str.format('Estimate time remaining: {}:{}:{}'.format(
            '{:02}'.format(hours), '{:02}'.format(minutes), '{:02}'.format(seconds)))
        return print_it

    def __create_rash_group(self):

        self.ldb = LotteryDatabase()
        fetch_a = self.ldb.execute(self.input_table)

        first_data_group = defaultdict(int)
        second_data_group = defaultdict(int)
        third_data_group = defaultdict(int)
        new_set = []

        for line in fetch_a:
            line = line[1:7]
            for num in line:
                if num in self.curr_game['groups']['first_group']:
                    new_set.append(1)
                elif num in self.curr_game['groups']['second_group']:
                    new_set.append(2)
                elif num in self.curr_game['groups']['third_group']:
                    new_set.append(3)

            count = Counter(new_set)

            if count[1] == 4:
                self.rash_one += 1
                for x, y in zip(line, new_set):
                    if y == 1:
                        first_data_group[x] += 1

            elif count[2] == 4:
                self.rash_two += 1
                for x, y in zip(line, new_set):
                    if y == 2:
                        second_data_group[x] += 1

            elif count[3] == 4:
                self.rash_three += 1
                for x, y in zip(line, new_set):
                    if y == 3:
                        third_data_group[x] += 1

            rash_group = []

            for num in line:
                if num in self.curr_game['groups']['first_group']:
                    d = first_data_group.get(num)
                    if d == "":
                        per = round(float(1 / self.rash_one) * 10, 2)
                    else:
                        per = round(float(d / self.rash_one) * 10, 2)

                    rash_group.append(per)
                elif num in self.curr_game['groups']['second_group']:
                    d = second_data_group.get(num)
                    if d == "":
                        per = round(float(1 / self.rash_two) * 10, 2)
                    else:
                        per = round(float(d / self.rash_two) * 10, 2)
                    rash_group.append(per)
                elif num in self.curr_game['groups']['third_group']:
                    d = third_data_group.get(num)
                    if d == "":
                        per = round(float(1 / self.rash_three) * 10, 2)
                    else:
                        per = round(float(d / self.rash_three) * 10, 2)

                    rash_group.append(per)

        self.ldb.__del__()

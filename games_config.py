VERSION = 'MYOPM v1.0'

CONFIG = {
    'poland_lotto': {
        'game': {
            'name': 'poland_lotto',
            'input_table': 'poland_lotto_649',
            'input_model': 'InputPolandLotto',
            'training_model': 'TrainingPolandLotto',
            'predict_model': 'PredictPolandLotto',
            'total_numbers': 49,
            'length': 6
        },
        'number_groups': {
            'game': 'poland_lotto',
            'first_group': '|'.join(map(str, [1, 2, 3, 4, 5, 6, 17, 18, 19, 20, 21, 33, 34, 35, 36, 37])),
            'second_group': '|'.join(map(str, [7, 8, 9, 10, 11, 12, 22, 23, 24, 25, 26, 38, 39, 40, 41, 42])),
            'third_group': '|'.join(map(str, [13, 14, 15, 16, 27, 28, 29, 30, 31, 32, 43, 44, 45, 46, 47, 48, 49]))
        },
        'alphabetic_groups': {
            'game': 'poland_lotto',
            'group_A': '|'.join(map(str, [1, 2, 3, 4, 5])),
            'group_B': '|'.join(map(str, [6, 7, 8, 9])),
            'group_C': '|'.join(map(str, [10, 11, 12, 13, 14, 15])),
            'group_D': '|'.join(map(str, [16, 17, 18, 19, 20])),
            'group_E': '|'.join(map(str, [21, 22, 23, 24, 25])),
            'group_F': '|'.join(map(str, [26, 27, 28, 29, 30])),
            'group_G': '|'.join(map(str, [31, 32, 33, 34, 35])),
            'group_H': '|'.join(map(str, [36, 37, 38, 39, 40])),
            'group_I': '|'.join(map(str, [41, 42, 43, 44, 45])),
            'group_J': '|'.join(map(str, [46, 47, 48, 49]))
        },
        'model_features': {
            'f1': {
                'name': 'number_map',
                'length': 49,
                'header': 'MAP_',
            },
            'f2': {
                'name': 'number_cycles',
                'length': 49,
                'header': 'CYCLE_',
            },
            'f3': {
                'name': 'in_last_draw',
                'length': 49,
                'header': 'IN_LAST_',
            },
            'f4': {
                'name': 'hot_numbers',
                'length': 49,
                'header': 'HOT_',
            },
            'f5': {
                'name': 'cold_numbers',
                'length': 49,
                'header': 'COLD_',
            },
            'f6': {
                'name': 'warm_numbers',
                'length': 49,
                'header': 'WARM_',
            },
            'f7': {
                'name': 'cool_numbers',
                'length': 49,
                'header': 'COOL_',
            },
            'f8': {
                'name': 'original_numbers',
                'length': 6,
                'header': 'ORIGINAL_',
            },
            'f9': {
                'name': 'rash_group',
                'length': 6,
                'header': 'RASH_',
            },
            'f10': {
                'name': 'alphabetic_group',
                'length': 60,
                'header': 'ALPHA_',
            },
        }
    },
    'poland_mini_lotto': {
        'game': {
            'name': 'poland_mini_lotto',
            'input_table': 'poland_mini_lotto_542',
            'input_model': 'InputPolandMiniLotto',
            'training_model': 'TrainingPolandMiniLotto',
            'predict_model': 'PredictPolandMiniLotto',
            'total_numbers': 42,
            'length': 5
        },
        'number_groups': {
            'first_group': '|'.join(map(str, [1, 2, 3, 4, 15, 16, 17, 18, 19, 29, 30, 31, 32, 33])),
            'second_group': '|'.join(map(str, [5, 6, 7, 8, 9, 20, 21, 22, 23, 24, 34, 35, 36, 37])),
            'third_group': '|'.join(map(str, [10, 11, 12, 13, 14, 25, 26, 27, 28, 38, 39, 40, 41, 42]))
        },
        'alphabetic_groups': {
            'group_A': '|'.join(map(str, [1, 2, 3, 4, 5])),
            'group_B': '|'.join(map(str, [6, 7, 8, 9, 10])),
            'group_C': '|'.join(map(str, [11, 12, 13, 14, 15])),
            'group_D': '|'.join(map(str, [16, 17, 18, 19, 20])),
            'group_E': '|'.join(map(str, [21, 22, 23, 24, 25])),
            'group_F': '|'.join(map(str, [26, 27, 28, 29, 30])),
            'group_G': '|'.join(map(str, [31, 32, 33, 34, 35])),
            'group_H': '|'.join(map(str, [36, 37, 38, 39, 40])),
            'group_I': '|'.join(map(str, [41, 42])),
            'group_J': ''
        },
        'model_features': {
            'f1': {
                'name': 'number_map',
                'length': 42,
                'header': 'MAP_',
            },
            'f2': {
                'name': 'number_cycles',
                'length': 42,
                'header': 'CYCLE_',
            },
            'f3': {
                'name': 'in_last_draw',
                'length': 42,
                'header': 'IN_LAST_',
            },
            'f4': {
                'name': 'hot_numbers',
                'length': 42,
                'header': 'HOT_',
            },
            'f5': {
                'name': 'cold_numbers',
                'length': 42,
                'header': 'COLD_',
            },
            'f6': {
                'name': 'warm_numbers',
                'length': 42,
                'header': 'WARM_',
            },
            'f7': {
                'name': 'cool_numbers',
                'length': 42,
                'header': 'COOL_',
            },
            'f8': {
                'name': 'original_numbers',
                'length': 5,
                'header': 'ORIGINAL_',
            },
            'f9': {
                'name': 'rash_group',
                'length': 5,
                'header': 'RASH_',
            },
            'f10': {
                'name': 'alphabetic_group',
                'length': 45,
                'header': 'ALPHA_',
            },
        }
    }
}


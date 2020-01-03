VERSION = 'MYOPM v1.0'

CONFIG = {
    'poland_lotto': {
        'game': {
            'game_name': 'poland_lotto',
            'game_table': 'poland_lotto_649',
            'total_numbers': 49,
            'game_len': 6
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
                'feature_name': 'number_map',
                'feature_length': 49,
                'feature_header': 'MAP',
            },
            'f2': {
                'feature_name': 'number_cycles',
                'feature_length': 49,
                'feature_header': 'CYCLE',
            },
            'f3': {
                'feature_name': 'in_last_draw',
                'feature_length': 49,
                'feature_header': 'IN_LAST',
            },
            'f4': {
                'feature_name': 'hot_numbers',
                'feature_length': 49,
                'feature_header': 'HOT',
            },
            'f5': {
                'feature_name': 'cold_numbers',
                'feature_length': 49,
                'feature_header': 'COLD',
            },
            'f6': {
                'feature_name': 'warm_numbers',
                'feature_length': 49,
                'feature_header': 'WARM',
            },
            'f7': {
                'feature_name': 'cool_numbers',
                'feature_length': 49,
                'feature_header': 'COOL',
            },
            'f8': {
                'feature_name': 'original_numbers',
                'feature_length': 6,
                'feature_header': 'ORIGINAL',
            },
            'f9': {
                'feature_name': 'rash_group',
                'feature_length': 6,
                'feature_header': 'RASH',
            },
            'f10': {
                'feature_name': 'alphabetic_group',
                'feature_length': 60,
                'feature_header': 'ALPHA',
            },
        }
    },
    'poland_mini_lotto': {
        'game': {
            'game_name': 'poland_mini_lotto',
            'game_table': 'poland_mini_lotto_542',
            'total_numbers': 42,
            'game_len': 5
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
                'feature_name': 'number_map',
                'feature_length': 42,
                'feature_header': 'MAP',
            },
            'f2': {
                'feature_name': 'number_cycles',
                'feature_length': 42,
                'feature_header': 'CYCLE',
            },
            'f3': {
                'feature_name': 'in_last_draw',
                'feature_length': 42,
                'feature_header': 'IN_LAST',
            },
            'f4': {
                'feature_name': 'hot_numbers',
                'feature_length': 42,
                'feature_header': 'HOT',
            },
            'f5': {
                'feature_name': 'cold_numbers',
                'feature_length': 42,
                'feature_header': 'COLD',
            },
            'f6': {
                'feature_name': 'warm_numbers',
                'feature_length': 42,
                'feature_header': 'WARM',
            },
            'f7': {
                'feature_name': 'cool_numbers',
                'feature_length': 42,
                'feature_header': 'COOL',
            },
            'f8': {
                'feature_name': 'original_numbers',
                'feature_length': 5,
                'feature_header': 'ORIGINAL',
            },
            'f9': {
                'feature_name': 'rash_group',
                'feature_length': 5,
                'feature_header': 'RASH',
            },
            'f10': {
                'feature_name': 'alphabetic_group',
                'feature_length': 45,
                'feature_header': 'ALPHA',
            },
        }
    }
}


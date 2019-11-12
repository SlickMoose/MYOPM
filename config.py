config = {
    'database': 'database/lotto.db',
    'user': 'user/settings.ini',
    'version': 'MYOPM v1.0',
    'games': {
        'lotto': {
            'name': 'Lotto 6 from 49',
            'database': 'original_649',
            'length': 49,
            'groups': {
                'first_group': [1, 2, 3, 4, 5, 6, 17, 18, 19, 20, 21, 33, 34, 35, 36, 37],
                'second_group': [7, 8, 9, 10, 11, 12, 22, 23, 24, 25, 26, 38, 39, 40, 41, 42],
                'third_group': [13, 14, 15, 16, 27, 28, 29, 30, 31, 32, 43, 44, 45, 46, 47, 48, 49]
            },
            'alphabetic_groups': {
                'group_A': [1, 2, 3, 4, 5],
                'group_B': [6, 7, 8, 9],
                'group_C': [10, 11, 12, 13, 14, 15],
                'group_D': [16, 17, 18, 19, 20],
                'group_E': [21, 22, 23, 24, 25],
                'group_F': [26, 27, 28, 29, 30],
                'group_G': [31, 32, 33, 34, 35],
                'group_H': [36, 37, 38, 39, 40],
                'group_I': [41, 42, 43, 44, 45],
                'group_J': [46, 47, 48, 49],
            },
            'features': {
                'number_map': {
                    'length': 49,
                    'header': 'MAP'
                },
                'number_cycles': {
                    'length': 49,
                    'header': 'CYCLE'
                },
                'in_last_draw': {
                    'length': 49,
                    'header': 'LAST'
                },
                'hot numbers': {
                    'length': 49,
                    'header': 'HOT'
                },
                'cold numbers': {
                    'length': 49,
                    'header': 'COLD'
                },
                'warm numbers': {
                    'length': 49,
                    'header': "WARM"
                },
                'cool numbers': {
                    'length': 49,
                    'header': 'COOL'
                },
                'original_numbers': {
                    'length': 6,
                    'header': 'ORIGINAL'
                },
                'rash_group': {
                    'length': 6,
                    'header': 'RASH'
                },
                'alphabetic_group': {
                    'length': 60,
                    'header': 'ALPHA'
                }
            }

        },
        'mini_lotto': {
            'name': 'Mini Lotto 5 from 42',
            'database': 'original_542',
            'length': 42,
            'groups': {
                'first_group': [1, 2, 3, 4, 15, 16, 17, 18, 19, 29, 30, 31, 32, 33],
                'second_group': [5, 6, 7, 8, 9, 20, 21, 22, 23, 24, 34, 35, 36, 37],
                'third_group': [10, 11, 12, 13, 14, 25, 26, 27, 28, 38, 39, 40, 41, 42]
            },
            'alphabetic_groups': {
                'group_A': [1, 2, 3, 4, 5],
                'group_B': [6, 7, 8, 9, 10],
                'group_C': [11, 12, 13, 14, 15],
                'group_D': [16, 17, 18, 19, 20],
                'group_E': [21, 22, 23, 24, 25],
                'group_F': [26, 27, 28, 29, 30],
                'group_G': [31, 32, 33, 34, 35],
                'group_H': [36, 37, 38, 39, 40],
                'group_I': [41, 42]
            },
            'features': {
                'number_map': {
                    'length': 42,
                    'header': 'MAP'
                },
                'number_cycles': {
                    'length': 42,
                    'header': 'CYCLE'
                },
                'in_last_draw': {
                    'length': 42,
                    'header': 'LAST'
                },
                'hot numbers': {
                    'length': 42,
                    'header': 'HOT'
                },
                'cold numbers': {
                    'length': 42,
                    'header': 'COLD'
                },
                'warm numbers': {
                    'length': 42,
                    'header': "WARM"
                },
                'cool numbers': {
                    'length': 42,
                    'header': 'COOL'
                },
                'original_numbers': {
                    'length': 5,
                    'header': 'ORIGINAL'
                },
                'rash_group': {
                    'length': 5,
                    'header': 'RASH'
                },
                'alphabetic_group': {
                    'length': 45,
                    'header': 'ALPHA'
                }
            }
        }

    }
}


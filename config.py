import os

# os/IO related
sub_dirs = ["logs", "evaluations", "dict", "weights"]
log_dir = 'logs'
run_name = 'test'
output_dir = 'output'

# data files
daily_prog_data = os.path.join("data", "DailyProgramData_04012015.csv")


# features:
"""
feature_type_map = {
    # time related features
     t_0: day of the week
     t_1: part of day (morning, noon, evening, night) in bucket of 6
     t_2: day of week and part of day
    # demographic related:
     from Dafna
    # genre related:
     g_0: current genre
     g_1: 2-gram
     g_2  3-gram
    # program related:
     
    # interactions:
     i_0: g_0 * t_0
     i_1: g_0 * t_1
     i_2: g_1 * t_0
     i_3: g_1 * t_1
     i_4: g_2 * t_0
     i_5: g_2 * t_1
     
 } 
"""
col_action = {
    'Program Genre': ['counter', 'g0'],
    'event_weekday': ['unique', 't0'],
    'part_of_day': ['unique', 't1'],
    'duration_bins': ['unique', 'p0'],
    'view_bins': ['unique', 'p1'],
    'completion_bins': ['unique', 'p2'],
    ('part_of_day', 'event_weekday'): ['interact', 't3'],
    ('Program Genre', 'event_weekday'): ['interact', 'i0'],
    ('Program Genre', 'part_of_day'): ['interact', 'i1'],
    ('prev_1_genre', 'Program Genre'): ['interact', 'g1'],
    ('prev_2_genre', 'prev_1_genre', 'Program Genre'): ['double_interact', 'g2']
}

thrash_holds = {
    'Program Genre': 30,
    ('part_of_day', 'event_weekday'): 30,
    ('Program Genre', 'event_weekday'): 30,
    ('Program Genre', 'part_of_day'): 30,
    ('prev_1_genre', 'Program Genre'): 30,
    ('prev_2_genre', 'prev_1_genre', 'Program Genre'): 30

}


# genres
# genere_map =

# raw data names
x_row_index = 'df_id'
x_device_id = 'Device ID'
demo_device_id = 'device_id'
x_label = 'genre'
# threshold for demographic features
min_amount_demo = 5
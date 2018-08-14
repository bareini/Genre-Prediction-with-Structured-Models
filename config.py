import os

# os/IO related
sub_dirs = ["logs", "evaluations", "dict", "weights"]
log_dir = 'logs'
run_name = 'test'
output_dir = 'output'
data_dir = 'data'
weights_folder = 'weights'
results_folder = 'evaluations'

# data files
daily_prog_data = os.path.join("data", "DailyProgramData_04012015.csv")
viewing_data_name = 'viewing.pkl'
demo_file_name = 'demographic_features.csv'
device_house_dict = 'dev_house_dict.pkl'
house_device_dict = 'house_dev_dict.pkl'
weights_file_name = os.path.join(weights_folder, 'my_weights.txt')
results_file_name = os.path.join(results_folder, 'my_results.txt')

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
     p_0: duration
     p_1: view duration
     p_2: program completion
     p_3: station number
    # interactions:
     i_0: g_0 * t_0
     i_1: g_0 * t_1
     i_2: g_1 * t_0
     i_3: g_1 * t_1
     i_4: g_2 * t_0
     i_5: g_2 * t_1
     
     # commons
     c1: p_3 * g_0
 } 
"""
col_action = {
    'Program Genre': ['counter', 'g0'],
    'event_weekday': ['unique', 't0'],
    'part_of_day': ['unique', 't1'],
    'duration_bins': ['unique', 'p0'],
    'view_bins': ['unique', 'p1'],
    'completion_bins': ['unique', 'p2'],
    'Station Number': ['counter', 'p3'],
    ('part_of_day', 'event_weekday'): ['interact', 't3'],
    ('event_weekday', 'Program Genre'): ['interact', 'i0'],
    ('part_of_day', 'Program Genre'): ['interact', 'i1'],
    ('prev_1_genre', 'Program Genre'): ['interact', 'g1'],
    ('prev_2_genre', 'prev_1_genre', 'Program Genre'): ['double_interact', 'g2'],
    ('Station Number', 'Program Genre'): ['interact', 'c1']
}
# 'Program Genre' must appear last in the feature name string
genere_cols = {

    'Program Genre': ['counter', 'g0'],
    ('event_weekday', 'Program Genre'): ['interact', 'i0'],
    ('part_of_day', 'Program Genre'): ['interact', 'i1'],
    ('prev_1_genre', 'Program Genre'): ['interact', 'g1'],
    ('prev_2_genre', 'prev_1_genre', 'Program Genre'): ['double_interact', 'g2'],
    ('Station Number', 'Program Genre'): ['interact', 'c1']
}

thresholds = {
    'Program Genre': 0,
    ('part_of_day', 'event_weekday'): 0,
    ('event_weekday', 'Program Genre'): 0,
    ('part_of_day', 'Program Genre'): 0,
    ('prev_1_genre', 'Program Genre'): 0,
    ('prev_2_genre', 'prev_1_genre', 'Program Genre'): 0,
    'Station Number': 0,
}


# genres
# genere_map =

# raw data names
x_row_index = 'df_id'
x_device_id = 'Device ID'
demo_device_id = 'device_id'
x_label = 'genre'
x_program_genre = 'Program Genre'
# threshold for demographic features
min_amount_demo = 0
household_id = 'Household ID'
station_genre = 'c1'
voter = 'Voter/Party'
station_num = 'Station Number'
train_threshold = 0.8

import os

# os/IO related
sub_dirs = ["logs", "evaluations", "dict", "weights"]
log_dir = 'logs'
run_name = 'test'
output_dir = 'output'
data_dir = 'data'
weights_folder = 'weights'
results_folder = 'evaluations'
dict_folder = 'dict'
models_folder = 'models'

# data files
daily_prog_data = os.path.join("data", "DailyProgramData_04012015.csv")
viewing_data_name = 'viewing_final.pkl.gz'
demo_file_name = 'demo.pkl.gz'
device_house_dict = 'dev_house_dict_new.pkl'
house_device_dict = 'house_dev_dict_new.pkl'
weights_file_name = os.path.join(weights_folder, 'my_weights.txt')
results_file_name = os.path.join(results_folder, 'my_results.txt')
household_list = 'hh_list.pkl'
clusters_df = 'viewing_clustered.pkl.gz'
full_viewing = 'viewing_full_advance_clustered.pkl.gz'

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
    ('duration_bins', 'Program Genre'): ['interact', 'p4'],  # NEW!
    ('part_of_day', 'event_weekday'): ['interact', 't3'],
    ('event_weekday', 'Program Genre'): ['interact', 'i0'],
    ('part_of_day', 'Program Genre'): ['interact', 'i1'],
    ('prev_1_genre', 'Program Genre'): ['interact', 'g1'],
    ('prev_2_genre', 'prev_1_genre', 'Program Genre'): ['double_interact', 'g2'],
    ('part_of_day', 'prev_1_genre', 'Program Genre'): ['double_interact', 'g3'], # NEW!
    ('part_of_day', 'Station Number', 'Program Genre'): ['double_interact', 'g4'],  # NEW!
    ('Station Number', 'Program Genre'): ['interact', 'c1']
}

no_genre_cols = {
    'event_weekday': ['unique', 't0'],
    'part_of_day': ['unique', 't1'],
    'duration_bins': ['unique', 'p0'],
    'view_bins': ['unique', 'p1'],
    'completion_bins': ['unique', 'p2'],
    'Station Number': ['counter', 'p3'],
    ('part_of_day', 'event_weekday'): ['interact', 't3'],
    ('part_of_day', 'Station Number', 'Program Genre'): ['double_interact', 'g4'],  # NEW!

}



# 'Program Genre' must appear last in the feature name string
genere_cols = {

    'Program Genre': ['counter', 'g0'],
    ('event_weekday', 'Program Genre'): ['interact', 'i0'],
    ('part_of_day', 'Program Genre'): ['interact', 'i1'],
    ('prev_1_genre', 'Program Genre'): ['interact', 'g1'],
    ('prev_2_genre', 'prev_1_genre', 'Program Genre'): ['double_interact', 'g2'],
    ('Station Number', 'Program Genre'): ['interact', 'c1'],
    ('duration_bins', 'Program Genre'): ['interact', 'p4'],   # NEW!
    ('part_of_day', 'prev_1_genre', 'Program Genre'): ['double_interact', 'g3'],  # NEW!
    ('part_of_day', 'Station Number', 'Program Genre'): ['double_interact', 'g4'],  # NEW!


}

All_cols = [
    no_genre_cols, genere_cols
]

advanced_household32 = {
    'gen_in_dev_hh_1': ['counter', 'h0'],
    ('gen_in_dev_hh_1', 'Program Genre'): ['interact', 'h1'],
    ('gen_in_dev_hh_1', 'part_of_day'): ['interact', 'h2'],
    ('event_weekday', 'gen_in_dev_hh_1'): ['interact', 'h3'],
    ('part_of_day', 'gen_in_dev_hh_1', 'Program Genre'): ['double_interact', 'h4']
}

advanced_pattern2 = {
    'gen_in_advance_1': ['counter', 'ad0'],
    ('gen_in_advance_1', 'Program Genre'): ['interact', 'ad1'],
    ('gen_in_advance_1', 'part_of_day'): ['interact', 'ad2'],
    ('event_weekday', 'gen_in_advance_1'): ['interact', 'h3'],
    ('part_of_day', 'gen_in_advance_1', 'Program Genre'): ['double_interact', 'ad4']
}

cluster_cols = {
    'program_genre_clustered': ['counter', 'ad0'],
    ('program_genre_clustered', 'Program Genre'): ['interact', 'ad1'],
    ('prev_1_genre_clustered', 'part_of_day'): ['interact', 'ad2'],
    ('event_weekday', 'program_genre_clustered'): ['interact', 'ad3'],
    ('prev_1_genre_clustered', 'program_genre_clustered', 'Program Genre'): ['double_interact', 'ad4'],
    ('prev_2_genre_clustered', 'prev_1_genre_clustered', 'program_genre_clustered'): ['double_interact', 'ad5']
    #  ('gen_in_dev_hh_1_clustered', 'program_genre_clustered'): ['interact', 'ad6'],
    #  ('gen_in_dev_hh_1_clustered', 'Program Genre'): ['interact', 'ad7']

}


thresholds = {
    'Program Genre': 10,
    ('part_of_day', 'event_weekday'): 3,
    ('event_weekday', 'Program Genre'): 3,
    ('part_of_day', 'Program Genre'): 3,
    ('prev_1_genre', 'Program Genre'): 3,
    ('prev_2_genre', 'prev_1_genre', 'Program Genre'): 2,
    ('duration_bins', 'Program Genre'): 3,
    ('part_of_day', 'prev_1_genre', 'Program Genre'): 0,
    'Station Number': 3,
    ('part_of_day', 'prev_1_genre', 'Program Genre'): 3,
    ('part_of_day', 'Station Number', 'Program Genre'): 5,
    ('Station Number', 'Program Genre'): 10,
    'gen_in_dev_hh_1': 10,
    ('gen_in_dev_hh_1', 'Program Genre'): 10,
    ('gen_in_dev_hh_1', 'part_of_day'): 10,
    ('event_weekday', 'gen_in_dev_hh_1'): 10,
    ('part_of_day', 'gen_in_dev_hh_1', 'Program Genre'): 10,
    'gen_in_advance_1': 5,
    ('gen_in_advance_1', 'Program Genre'): 5,
    ('gen_in_advance_1', 'part_of_day'): 5,
    ('event_weekday', 'gen_in_advance_1'): 5,
    ('gen_in_advance_1_prev1', 'gen_in_advance_1', 'Program Genre'): 5,
    ('gen_in_advance_1_prev2', 'gen_in_advance_1_prev1', 'Program Genre'): 5

}

model_types = ['basic', 'Advanced']

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
genre_prefixes = ['g0', 'g1', 'g2','g3', 'i0','c1','p4']
station_time_genre = 'g4'
part_of_day_genre = 'i1'
part_of_day = 'part_of_day'
num_of_top_k = 5
x_household_id = 'household_id'
x_clustered_genre = 'program_genre_clustered'


num_of_iters = 3
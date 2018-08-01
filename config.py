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
# genres
# genere_map =

# raw data names
x_row_index = 'df_id'
x_device_id = 'Device ID'
demo_device_id = 'device_id'
x_label = 'genre'

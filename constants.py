
#metrics for analyzing pitcher performance
pitcher_metrics = ['IP','TBF','WHIP','ERA', 'FIP', 'K%', 'BB%', 'K-BB%']

# Global dictionary for pitch type descriptions
dict_pitch = {
    "FF": "Four-Seam Fastball",
    "SL": "Slider",
    "CH": "Changeup",
    "CU": "Curveball",
    "SI": "Sinker",
    "FC": "Cutter",
    "KN": "Knuckleball",
    "FS": "Splitter",
    "EP": "Eephus",
    # Add other pitch types as needed
}

#metrics for each pitch thrown by a pitcher
pitch_stats_dict = {
    'pitch': {'table_header': '$\\bf{Count}$', 'format': '.0f'},
    'release_speed': {'table_header': '$\\bf{Velocity}$', 'format': '.1f'},
    'pfx_z': {'table_header': '$\\bf{iVB}$', 'format': '.1f'},
    'pfx_x': {'table_header': '$\\bf{HB}$', 'format': '.1f'},
    'release_spin_rate': {'table_header': '$\\bf{Spin}$', 'format': '.0f'},
    'release_pos_x': {'table_header': '$\\bf{hRel}$', 'format': '.1f'},
    'release_pos_z': {'table_header': '$\\bf{vRel}$', 'format': '.1f'},
    'release_extension': {'table_header': '$\\bf{Ext.}$', 'format': '.1f'},
    'xwoba': {'table_header': '$\\bf{xwOBA}$', 'format': '.3f'},
    'pitch_usage': {'table_header': '$\\bf{Pitch\\%}$', 'format': '.1%'},
    'whiff_rate': {'table_header': '$\\bf{Whiff\\%}$', 'format': '.1%'},
    'in_zone_rate': {'table_header': '$\\bf{Zone\\%}$', 'format': '.1%'},
    'chase_rate': {'table_header': '$\\bf{Chase\\%}$', 'format': '.1%'},
    'delta_run_exp_per_100': {'table_header': '$\\bf{RV\\/100}$', 'format': '.1f'}
}

# so that I can filter the incomplete pitch data (spring training games have NAN values for valuable attributes)
OPENING_DAY = {
    2024: "2024-03-28",
    2023: "2023-03-30",
    2022: "2022-04-07",
    2021: "2021-04-01",
    2020: "2020-07-23",
    2019: "2019-03-28",
    2018: "2018-03-29",
    2017: "2017-04-02",
    2016: "2016-04-03",
    2015: "2015-04-05"
}



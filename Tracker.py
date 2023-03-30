import numpy as np
import pandas as pd
from statistics import mean

class Tracker: # currently tracking number of triage waiters
    def __init__(self, warm_up) -> None:
        self.warm_up = warm_up
        # some place holders to track number of waiters by points in time
        self.env_time_all = []
        self.waiters = {
            'triage': [],
            'cubicle': [],
            'ae_doc': [],
            'miu_doc': []
        }
        self.waiters_all = {
            'triage': [],
            'cubicle': [],
            'ae_doc': [],
            'miu_doc': []
        }
        # empty df to hold patient level details, including time in system, priority etc
        self.results_df = pd.DataFrame()
        self.results_df["P_ID"] = []
        self.results_df["Priority"] = []
        self.results_df["TriageOutcome"] = []
        self.results_df["TimeInSystem"] = []
        self.results_df["Admitted"] = []
        self.results_df.set_index("P_ID", inplace=True)

    def plot_data(self, env_time, type):
        if env_time > self.warm_up:
            self.waiters_all[type].append(len(self.waiters[type]))
            self.env_time_all.append(env_time)

    def mean_priority_wait(self):
        self.priority_means = {}
        for i in range(1, 6):
            try:
                self.priority_means["Priority{0}".format(i)] = mean(self.results_df[self.results_df['Priority'] == i]['TimeInSystem'])
            except:
                self.priority_means["Priority{0}".format(i)] = np.NaN

    def priority_count(self):
        self.priority_counts = {}
        for i in range(1, 6):
            try:
                self.priority_counts["Priority{0}".format(i)] = len(self.results_df[self.results_df['Priority'] == i]['TimeInSystem'])
            except:
                self.priority_counts["Priority{0}".format(i)] = 0
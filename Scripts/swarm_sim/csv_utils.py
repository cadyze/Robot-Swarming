import numpy as np
import pandas as pd


class CSVUtils:
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)

        # Filter and receive timestep values
        self.m_values = []
        self.m_sample_values = []
        self.sample_size = 0

        # Check if 'Timesteps' column exists in the CSV
        if 'Timesteps' in self.df.columns:
            self.m_values = self.df['Timesteps'].values
            self.m_values = self.m_values[self.m_values > 10]

    def get_mean(self):
        """Returns the mean of the 'Timesteps' column."""
        return self.m_values.mean()

    def generate_new_sample(self, sample_size):
        """Returns a random sample of size 'sample_size' from the 'Timesteps' column."""
        self.sample_size = sample_size
        if len(self.m_values) < sample_size:
            self.m_sample_values = self.m_values
        else:
            self.m_sample_values = np.random.choice(self.m_values, size=sample_size, replace=False)

    def get_sampled_mean(self):
        """Returns the mean of a random sample of size 'sample_size' from the 'Timesteps' column."""
        if len(self.m_values) < self.sample_size:
            return self.get_mean()
        else:
            return self.m_sample_values.mean()

    def get_variance(self):
        """Returns the variance of the 'Timesteps' column."""
        return self.m_values.std()

    def get_sampled_variance(self):
        """Returns the variance of a random sample of size 'sample_size' from the 'Timesteps' column."""
        if len(self.m_values) < self.sample_size:
            return self.get_variance()
        else:
            return self.m_sample_values.std()

    def get_count(self):
        """Returns the count of the 'Timesteps' column."""
        return len(self.m_values)

    def get_wait_percent(self):
        """Returns the percentage of steps waited."""
        total_timesteps = self.df['Timesteps'].sum()
        total_time_waited = self.df['Steps Waited'].sum()
        return total_time_waited / total_timesteps * 100

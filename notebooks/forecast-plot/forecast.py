#!/usr/bin/env python3

"""
Description: This script contains helper functions for collecting National 
             Water Model timeseries data using the BiGQuery API, located 
             at https://nwm-api.ciroh.org/

Author(s): Tony Castronova <acastronova@cuahsi.org>
"""

import io
import pandas
import requests
from enum import Enum
import concurrent.futures
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
from typing import List

# use the notebook version of TQDM if running in Jupyter
# otherwise use the regular version.
try:
    from tqdm.notebook import tqdm as tqdm
except ImportError:
    from tqdm import tqdm



class ForecastTypes(Enum):
    """
    CONUS Streamflow Forecast data available via nwm-api.ciroh.org
    https://nwm-api.ciroh.org/docs#/default/analysis_assim_analysis_assim_get
    
    - Short-Range Forecast: 18-hour deterministic (single value) forecast
    - Medium-Range Forecast: Six-member ensemble forecasts out to 10 days (member 1) and 8.5 days (members 2-6) initialized by standard analysis and forced by a time-lagged ensemble of GFS data
    - Long-Range Forecast: 30-day four-member ensemble forecast
    """
    
    SHORT = 'short_range'
    MEDIUM = 'medium_range'
    LONG = 'long_range'


    
class Forecasts():
    def __init__(self, api_key, api_url='https://nwm-api.ciroh.org'):
        self.url = f'{api_url}/forecast'
        self.header =  {
            'x-api-key': api_key
        }
        self.df = None

        self.ENSEMBLES = {
            ForecastTypes.SHORT: [0], 
            ForecastTypes.MEDIUM: [0, 1, 2, 3, 4, 5],
            ForecastTypes.LONG: [0, 1, 2, 3]
        }
        
    def filter_forecast_ensembles(self, forecast_type: ForecastTypes, ensembles:List[str] = ['all']):
        """
        Filters the ensembles, returning only those that exist in the given forecast type.
        
        :param forecast_type: ForecastType - The type of forecast to filter ensembles for.
        :param ensembles: list or str - A list of user-defined ensembles to filter, or 'all' to return all ensembles.
        :return: A list of valid ensembles for the specified forecast type.
        """
        if forecast_type not in self.ENSEMBLES:
            raise ValueError(f"Invalid forecast type: {forecast_type}")
    
        valid_ensembles = self.ENSEMBLES[forecast_type]
        
        if 'all' in ensembles:
            return valid_ensembles  # Return all IDs for the given forecast type
        else:
            return [i for i in ensembles if int(i) in valid_ensembles]

    
    def fetch_url(self, params):
        try:
            response = requests.get(self.url,
                                    params=params,
                                    headers=self.header)
            
            # Raise an exception for HTTP errors
            response.raise_for_status()  
            return response
            
        except requests.exceptions.RequestException as e:
            return f"Error fetching {self.url}: {e}"

    def fetch_async(self, params_list):

        results = []
        errors = []
        
        # Use ThreadPoolExecutor to make concurrent GET requests
        # TQDM is used to provide a nice looking progress bar
        with ThreadPoolExecutor(max_workers=5) as executor:
            
            # Submit all URLs to the executor
            future_to_url = {executor.submit(self.fetch_url, param): param for param in params_list}
            
            # Process the results as they complete
            for future in tqdm(concurrent.futures.as_completed(future_to_url),
                               total=len(future_to_url),
                               desc="Fetching Forecast Data",
                               unit="url",
                               colour="green",  
                               dynamic_ncols=True):
                url = future_to_url[future]
                try:
                    res = future.result()

                    # attempt to get the status code.
                    # if one is not returned, we should log 
                    # it as an error.
                    status_code = res.status_code

                    # otherwise, the 
                    results.append(res)
                    
                except Exception as e:
                    errors.append(f"Exception for {url}: {e}")
            
            return results, errors
            
        
            
    def collect_forecasts(self,
                          comids, 
                          forecast_type,
                          reference_times,
                          ensembles='all'):
        """
        :param ensembles: list or str - A list of user-defined ensembles to filter, or 'all' to return all ensembles.
        """

        # get ensembles
        valid_ensembles = self.filter_forecast_ensembles(forecast_type, ensembles=ensembles)
        
        # build a parameters to query
        params = [
            {'comids': ','.join(map(str, comids)), 
             'forecast_type': forecast_type.value,
             'reference_time': reftime,
             'ensemble': ','.join(map(str, valid_ensembles)),
             'output_format': 'csv'}
            for reftime in reference_times
        ]

        # query the api asynchronously with the parameters defined above 
        responses, errors = self.fetch_async(params)
        # TODO: do something if we encounter an error.

        # filter out only the successful responses and 
        # convert them into a single pandas dataframe
        successful_responses = [resp for resp in responses if resp.status_code == 200]
        dfs = [pandas.read_csv(io.StringIO(res.text), sep=',') for res in successful_responses]

        if len(dfs) > 1:
            df = pandas.concat(dfs, ignore_index=True)  
        else:
            df = dfs[0]

        # clean datetime columns and return
        df.time = pandas.to_datetime(df.time)
        df.reference_time = pandas.to_datetime(df.reference_time)

        self.df = df

    def plot(self, comid, plot_type='series', xlabel='Time', ylabel='Streamflow'):
        
        if self.df is None:
            print('No forecast data to plot. Run "collect_forecasts" to collect data')
            return None
            
        df = self.df[self.df.feature_id == int(comid)]
        
        fig, ax = plt.subplots(figsize=(10, 5))
        
        if plot_type == 'series':
            self.__plot_series(df, ax)
        elif plot_type == 'iqr':
            self.__plot_iqr(df, ax)
        else:
            print(f'Unrecognized plot type: {plot_type}')
            return None

        ax.grid(True)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        plt.xticks(rotation=25) 
        
        return plt, ax

    def __plot_series(self, df, ax):
        
        
        # Group by 'reference_time' and plot each group on the same axis
        for reference_time, group in df.groupby('reference_time'):
            group.plot(x='time', y='streamflow', ax=ax, label=str(reference_time), legend=False)
            
    def __plot_iqr(self, df, ax):
        
        iqr = df.groupby(df.time)['streamflow'].quantile([0.25, 0.75])
        iqr = iqr.reset_index()
        iqr = iqr.rename(columns={'level_1': 'quantile'})
        
        df_pivot = iqr.pivot(index='time', columns='quantile', values='streamflow')
        df_pivot.index = pandas.to_datetime(df_pivot.index)

        ax.fill_between(df_pivot.index, df_pivot[0.25], df_pivot[0.75], color='blue', alpha=0.3);


            
            


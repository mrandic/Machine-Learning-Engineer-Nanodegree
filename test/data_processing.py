import pandas as pd
import numpy as np
from dateutil.parser import parse


def processHubwayTripsData(hubway_trips_df):
    hubway_trips_df['start_date'] = hubway_trips_df['start_date'].apply(lambda x: parse(x))
    hubway_trips_df['year_start'] = hubway_trips_df['start_date'].apply(lambda x: x.year)
    hubway_trips_df['month_start'] = hubway_trips_df['start_date'].apply(lambda x: x.month)
    hubway_trips_df['weekday_start'] = hubway_trips_df['start_date'].apply(lambda x: x.dayofweek)
    hubway_trips_df['day_start'] = hubway_trips_df['start_date'].apply(lambda x: x.day)
    hubway_trips_df['hour_start'] = hubway_trips_df['start_date'].apply(lambda x: x.hour)
    hubway_trips_df = hubway_trips_df.rename(columns={'status': 'trip_status'})
    return hubway_trips_df


def mapFrequentPostalCodeToGPSData():
    dict = {'zip_code': ["'02118", "'02139", "'02215", "'02116", "'02115", "'02138", "'02114", "'02143", "'02113", "'02134" ],
            'zip_code_lat': [42.3407, 42.3643, 42.3476, 42.3514, 42.3480, 42.34733, 42.36033, 42.38371, 42.36285, 42.35595 ],
            'zip_code_lng': [-71.0708, -71.1022, -71.1009, -71.0776, -71.0885, -71.16867, -71.06732, -71.10213, -71.05518, -71.13411  ]
           }
    return pd.DataFrame(data=dict)


def createMasterDataSet(hubway_trips_df, hubway_stations_df, weather_df, zip_code_gps_df):
    hubway_trips_df = processHubwayTripsData(hubway_trips_df)
    master_df = pd.merge(hubway_trips_df, hubway_stations_df, how='left', left_on='strt_statn', right_on='id')
    master_df = master_df.rename(columns={'id': 'id_start', 'terminal': 'terminal_start', 'station': 'station_start',
                                          'municipal': 'municipal_start', 'lat': 'lat_start', 'lng': 'lng_start',
                                          'status': 'status_start'})
    master_df = pd.merge(master_df, hubway_stations_df, how='left', left_on='end_statn', right_on='id')
    master_df = master_df.rename(
        columns={'id': 'id_end', 'terminal': 'terminal_end', 'station': 'station_end', 'municipal': 'municipal_end',
                 'lat': 'lat_end', 'lng': 'lng_end', 'status': 'status_end'})
    master_df = pd.merge(master_df, weather_df, how='left', left_on=['year_start', 'month_start', 'day_start'],
                         right_on=['Year', 'Month', 'Day'])
    master_df = pd.merge(master_df, zip_code_gps_df, how='left', left_on=['zip_code'], right_on=['zip_code'])

    return master_df


def importData():
    hubway_stations_df = pd.read_csv('hubway_stations.csv', sep=',').sort_values(['station'], ascending=True)
    hubway_trips_df = pd.read_csv('hubway_trips.csv', sep=',')
    weather_df = pd.read_csv('boston_weather.csv', sep=',')
    zip_code_gps_df = mapFrequentPostalCodeToGPSData()

    return hubway_trips_df, hubway_stations_df, weather_df, zip_code_gps_df


def createFeatures(master_df):
    master_df['same_st_flg'] = np.where(master_df['strt_statn'] == master_df['end_statn'], 1, 0)
    master_df['age'] = master_df[(master_df['subsc_type'] == 'Registered')]['year_start'] - \
                       master_df[(master_df['subsc_type'] == 'Registered')]['birth_date']

    bins = [0, 2, 4, 6, 8, np.inf]
    names = ['0-2', '2-4', '4-6', '6-8', '8+']
    master_df['Avg Visibility Range (mi)'] = pd.cut(master_df['Avg Visibility (mi)'], bins, labels=names)

    bins = [20, 40, 60, 80, np.inf]
    names = ['20-40', '40-60', '60-80', '80+']
    master_df['Avg Temp Range (F)'] = pd.cut(master_df['Avg Temp (F)'], bins, labels=names)

    bins = [20, 40, 60, 80, np.inf]
    names = ['20-40', '40-60', '60-80', '80+']
    master_df['Avg Humidity Range (%)'] = pd.cut(master_df['Avg Humidity (%)'], bins, labels=names)

    bins = [0, 5, 10, 15, np.inf]
    names = ['0-5', '5-10', '10-15', '15+']
    master_df['Avg Wind Range (mph)'] = pd.cut(master_df['Avg Wind (mph)'], bins, labels=names)

    bins = [0, 20, 40, 60, np.inf]
    names = ['0-20', '20-40', '40-60', '60+']
    master_df['Avg Dew Point Range (F)'] = pd.cut(master_df['Avg Dew Point (F)'], bins, labels=names)

    bins = [0, 20, 40, 60, np.inf]
    names = ['0-20', '20-40', '40-60', '60+']
    master_df['Age Range'] = pd.cut(master_df[(master_df['subsc_type'] == 'Registered')]['age'], bins, labels=names)

    bike_agg = master_df[['bike_nr', 'seq_id', 'duration']].groupby(by=['bike_nr']).agg(
        bike_use_cnt=('seq_id', 'count'), bike_ride_duration_avg=('duration', 'mean')).sort_values(["bike_use_cnt"],
                                                                                                   ascending=(
                                                                                                       False)).reset_index()
    master_df = pd.merge(master_df, bike_agg, how='left', left_on=['bike_nr'], right_on=['bike_nr'])

    bins = [0, 500, 1000, 1500, np.inf]
    names = ['0-500', '500-1000', '1000-1500', '1500+']
    master_df['Bike Use Range'] = pd.cut(master_df['bike_use_cnt'], bins, labels=names)

    bins = [500, 1000, 1500, np.inf]
    names = ['500-1000', '1000-1500', '1500+']
    master_df['Bike Avg Time Use Range'] = pd.cut(master_df['bike_ride_duration_avg'], bins, labels=names)

    master_df = master_df[(master_df["duration"] > 0) & (master_df["duration"] <= 3000)]

    return master_df


def renameColumns(feature_set):
    feature_set = feature_set.rename(
    columns={'lat_start': 'latitude',
             'lng_start': 'longitude',
             'year_start': 'year',
             'month_start': 'month',
             'weekday_start': 'weekday',
             'day_start': 'day',
             'hour_start': 'hour',
             'municipal_start': 'staton_municipality',
             'status_start': 'station_status',
             'Bike Use Range': 'bike_freq_use_range',
             'Bike Avg Time Use Range': 'bike_avg_dur_range',
             'Avg Temp (F)': 'avg_tmp_f',
             'Avg Dew Point (F)': 'avg_dew_point_f',
             'Avg Humidity (%)': 'avg_humidity_pct',
             'Avg Sea Level Press (in)': 'avg_sea_level_press_in',
             'Avg Visibility (mi)': 'avg_visibility_mi',
             'Avg Wind (mph)': 'avg_wind_mph',
             'Snowfall (in)': 'sbowfall_in',
             'Precip (in)': 'precip_in',
             'Events': 'weather_event'
            })
    return feature_set


def featureSubset(master_df):
    feature_set = master_df[[
    'municipal_start',
    'lat_start',
    'lng_start',
    'status_start',
    'trip_status',
    'year_start',
    'month_start',
    'weekday_start',
    'day_start',
    'hour_start',
    'subsc_type',
    'zip_code',
    'gender',
    'age',
    'Bike Use Range',
    'Bike Avg Time Use Range',
    'Avg Temp (F)',
    'Avg Dew Point (F)',
    'Avg Humidity (%)',
    'Avg Sea Level Press (in)',
    'Avg Visibility (mi)',
    'Avg Wind (mph)',
    'Snowfall (in)',
    'Precip (in)',
    'Events',
    'duration'
    ]]
    return feature_set


def setFeatureCategoryType(feature_set):
    feature_set["bike_freq_use_range"] = feature_set["bike_freq_use_range"].astype('category')
    feature_set["bike_avg_dur_range"] = feature_set["bike_avg_dur_range"].astype('category')
    feature_set["staton_municipality"] = feature_set["staton_municipality"].astype('category')
    feature_set["station_status"] = feature_set["station_status"].astype('category')
    feature_set["trip_status"] = feature_set["trip_status"].astype('category')
    feature_set["subsc_type"] = feature_set["subsc_type"].astype('category')
    feature_set["zip_code"] = feature_set["zip_code"].astype('category')
    feature_set["gender"] = feature_set["gender"].astype('category')
    feature_set["weather_event"] = feature_set["weather_event"].astype('category')

    return feature_set
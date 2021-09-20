import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plotUserTypeStackData(master_df, title, group_var, variable, labels, label_x, label_y):
    month_use_counts = master_df[[group_var, 'subsc_type', 'seq_id', 'duration']].groupby(
        by=[group_var, 'subsc_type']).agg(use_cnt=('seq_id', 'count'),
                                          duration_avg=('duration', 'mean')).reset_index().sort_values(
        [group_var, 'subsc_type'], ascending=(True, True))
    reg_pct = month_use_counts[(month_use_counts['subsc_type'] == 'Registered')][variable]
    cas_pct = month_use_counts[(month_use_counts['subsc_type'] == 'Casual')][variable]

    width = 0.8

    fig, ax = plt.subplots()

    ax.bar(labels, reg_pct, width, label='Registered')
    ax.bar(labels, cas_pct, width, bottom=reg_pct,
           label='Casual')

    ax.set_ylabel(label_y)
    ax.set_xlabel(label_x)
    ax.set_title(title)
    ax.legend()

    # Matplotlib idiom to reverse legend entries
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(reversed(handles), reversed(labels))

    plt.show()


def plotUserGenderStackData(master_df, title, group_var, variable, labels, label_x, label_y):
    month_use_counts = master_df[[group_var, 'subsc_type', 'gender', 'seq_id', 'duration']].groupby(
        by=[group_var, 'subsc_type', 'gender']).agg(use_cnt=('seq_id', 'count'),
                                                    duration_avg=('duration', 'mean')).reset_index().sort_values(
        [group_var, 'gender'], ascending=(True, True))
    reg_pct = \
    month_use_counts[(month_use_counts['subsc_type'] == 'Registered') & (month_use_counts['gender'] == 'Male')][
        variable]
    cas_pct = \
    month_use_counts[(month_use_counts['subsc_type'] == 'Registered') & (month_use_counts['gender'] == 'Female')][
        variable]

    width = 0.8  # the width of the bars: can also be len(x) sequence

    fig, ax = plt.subplots()

    ax.bar(labels, reg_pct, width, label='Male')
    ax.bar(labels, cas_pct, width, bottom=reg_pct,
           label='Female')

    ax.set_ylabel(label_y)
    ax.set_xlabel(label_x)
    ax.set_title(title)
    ax.legend()

    # Matplotlib idiom to reverse legend entries
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(reversed(handles), reversed(labels))

    plt.show()


def plotSpatialDataClusterStations(feature_set, hubway_stations_df):
    city_map = plt.imread('boston_map.png')
    BBox = ((hubway_stations_df.lng.min(), hubway_stations_df.lng.max(), hubway_stations_df.lat.min(),
             hubway_stations_df.lat.max()))

    fig, ax = plt.subplots(figsize=(8, 7))
    clusters = np.unique(feature_set['location_cluster'])
    for i in clusters:
        ax.scatter(feature_set[feature_set['location_cluster'] == i].longitude,
                   feature_set[feature_set['location_cluster'] == i].latitude, zorder=1, alpha=1.0, s=30, label=i)

    ax.set_title('Bike Station Locations Grouped in Clusters')
    ax.set_xlim(BBox[0], BBox[1])
    ax.set_ylim(BBox[2], BBox[3])
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.imshow(city_map, zorder=0, extent=BBox, aspect='equal')
    labels = clusters
    plt.legend(labels)


def plotSpatialDataAllStations(master_df, hubway_stations_df):
    city_map = plt.imread('boston_map.png')
    BBox = ((hubway_stations_df.lng.min(), hubway_stations_df.lng.max(), hubway_stations_df.lat.min(), hubway_stations_df.lat.max()))
    most_used_stations = master_df[['strt_statn', 'lat_start', 'lng_start', 'seq_id']].groupby(by=['strt_statn', 'lat_start', 'lng_start']).agg(use_cnt=('seq_id', 'count')).sort_values(["use_cnt"], ascending = (False)).reset_index().head(10)
    least_used_stations = master_df[['strt_statn', 'lat_start', 'lng_start', 'seq_id']].groupby(by=['strt_statn', 'lat_start', 'lng_start']).agg(use_cnt=('seq_id', 'count')).sort_values(["use_cnt"], ascending = (True)).reset_index().head(10)
    fig, ax = plt.subplots(figsize = (8,7))
    ax.scatter(hubway_stations_df.lng, hubway_stations_df.lat, zorder=1, alpha= 1.0, c='b', s=30)
    ax.scatter(most_used_stations.lng_start, most_used_stations.lat_start, zorder=1, alpha= 1.0, c='g', s=30)
    ax.scatter(least_used_stations.lng_start, least_used_stations.lat_start, zorder=1, alpha= 1.0, c='r', s=30)
    ax.set_title('Bike Station Locations Grouped by Use Frequency')
    ax.set_xlim(BBox[0],BBox[1])
    ax.set_ylim(BBox[2],BBox[3])
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.imshow(city_map, zorder=0, extent = BBox, aspect= 'equal')
    labels = ["Regular", "Most Used", "Least Used"]
    plt.legend(labels)


def plotSpatialDataStationInteractions(master_df, hubway_stations_df):
    city_map = plt.imread('boston_map.png')
    BBox = ((hubway_stations_df.lng.min(), hubway_stations_df.lng.max(), hubway_stations_df.lat.min(),
             hubway_stations_df.lat.max()))
    stat_interactions = master_df[
        ['strt_statn', 'end_statn', 'lat_start', 'lng_start', 'lat_end', 'lng_end', 'seq_id', 'duration']].groupby(
        by=['strt_statn', 'end_statn', 'lat_start', 'lng_start', 'lat_end', 'lng_end']).agg(use_cnt=('seq_id', 'count'),
                                                                                            duration_avg=('duration',
                                                                                                          'mean')).sort_values(
        ["use_cnt"], ascending=(False)).reset_index().head(30)
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(stat_interactions.lng_start, stat_interactions.lat_start, zorder=1, alpha=0.3, c='b', s=100)
    ax.scatter(stat_interactions.lng_end, stat_interactions.lat_end, zorder=1, alpha=0.3, c='b', s=100)

    ax.set_title('Stations With The Most Frequent Interactions')
    ax.set_xlim(BBox[0], BBox[1])
    ax.set_ylim(BBox[2], BBox[3])
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.imshow(city_map, zorder=0, extent=BBox, aspect='equal')

    X_coords = [stat_interactions.lng_start, stat_interactions.lng_end]
    Y_coords = [stat_interactions.lat_start, stat_interactions.lat_end]
    plt.plot(X_coords, Y_coords, color='red')

    plt.show()


def plotSpatialDataUserZIPCodes(zip_code_gps_df):
    city_map = plt.imread('boston_map.png')
    BBox = ((zip_code_gps_df.zip_code_lng.min(), zip_code_gps_df.zip_code_lng.max(), zip_code_gps_df.zip_code_lat.min(), zip_code_gps_df.zip_code_lat.max()))
    fig, ax = plt.subplots(figsize = (8,7))
    ax.scatter(zip_code_gps_df.zip_code_lng, zip_code_gps_df.zip_code_lat, zorder=1, alpha= 1.0, c='b', s=30)
    ax.set_title('User Residence Areas With The Highest Frequency of Bike Use')
    ax.set_xlim(BBox[0],BBox[1])
    ax.set_ylim(BBox[2],BBox[3])
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.imshow(city_map, zorder=0, extent = BBox, aspect= 'equal')
    labels = ["User Residence Area"]
    plt.legend(labels)


def plotSpatialDataUserZIPCodeInteractions(master_df):
    zipcode_interactions = master_df[
        ['zip_code', 'id_start', 'lat_start', 'lng_start', 'zip_code_lat', 'zip_code_lng', 'seq_id']].groupby(
        by=['zip_code', 'id_start', 'lat_start', 'lng_start', 'zip_code_lat', 'zip_code_lng']).agg(
        use_cnt=('seq_id', 'count')).sort_values(["use_cnt"], ascending=(False)).reset_index().head(4)
    city_map = plt.imread('boston_map.png')
    BBox = ((
    zipcode_interactions.lng_start.min(), zipcode_interactions.lng_start.max(), zipcode_interactions.lat_start.min(),
    zipcode_interactions.lat_start.max()))
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(zipcode_interactions.zip_code_lng, zipcode_interactions.zip_code_lat, zorder=1, alpha=0.3, c='b', s=100)
    ax.scatter(zipcode_interactions.lng_start, zipcode_interactions.lat_start, zorder=1, alpha=0.3, c='b', s=100)

    ax.set_title('User Residence Area With The Highest Station Interaction')
    ax.set_xlim(BBox[0], BBox[1])
    ax.set_ylim(BBox[2], BBox[3])
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.imshow(city_map, zorder=0, extent=BBox, aspect='equal')

    X_coords = [zipcode_interactions.zip_code_lng, zipcode_interactions.lng_start]
    Y_coords = [zipcode_interactions.zip_code_lat, zipcode_interactions.lat_start]
    plt.plot(X_coords, Y_coords, color='red')

    plt.show()


def plotHistDistDuration(master_df, municipality):
    plt.hist(master_df[(master_df['municipal_start'] == municipality) & (master_df['subsc_type'] == 'Registered')][
                 'duration'], range=(0, 3000), bins=20, alpha=0.5)
    plt.hist(
        master_df[(master_df['municipal_start'] == municipality) & (master_df['subsc_type'] == 'Casual')]['duration'],
        range=(0, 3000), bins=20, alpha=0.5)
    plt.title('Frequency Distribution of Trip Duration for ' + municipality)
    plt.xlabel('Duration (s)')
    plt.ylabel('Number of Trips')

    labels = ["Registered", "Casual"]
    plt.legend(labels)
    plt.show()


def plotHistDistVar(master_df, title, xlabel, ylabel, var, start, end, bins_cnt):
    plt.hist(master_df[var],range = (start, end), bins = bins_cnt)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()




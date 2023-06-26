import pickle
import pandas as pd
import numpy as np
import sys

def read_data(filename):
    categorical = ['PULocationID', 'DOLocationID']
    
    print(filename + '\n')

    df = pd.read_parquet(filename)
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df

def main(year = 0000, month = 0):
    with open('model.bin', 'rb') as f_in:
        dv, model = pickle.load(f_in)

    categorical = ['PULocationID', 'DOLocationID']

    df = read_data(f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet')

    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = model.predict(X_val)

    # Calculating the standard deviation
    np.std(y_pred)

    # Mean predicted duration
    import statistics
    print(statistics.mean(y_pred))
    print
    print

    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')

    df.head()

    df_result = df[['duration', 'ride_id']]
    print(df_result.describe())

    # df_result.to_parquet(
    #     "resulting_dataframe.parquet",
    #     engine="pyarrow",
    #     compression=None,
    #     index=False
    # )

if __name__ == "__main__":
    year = int(sys.argv[1])
    month = int(sys.argv[2])
    main(year = year, month = month)
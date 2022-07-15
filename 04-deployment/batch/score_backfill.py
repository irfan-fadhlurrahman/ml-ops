from datetime import datetime
from dateutil.relativedelta import relativedelta

from prefect import flow

import score


@flow
def ride_duration_prediction_backfill():
    start_date = datetime(year=2021, month=3, day=1)
    end_date = datetime(year=2022, month=4, day=1)

    d = start_date

    while d <= end_date:
        score.ride_duration_prediction(
            taxi_type='green',
            run_id='d3b600b0d89e4e95b27b5e333e4d3c01',
            run_date=d
        )

        d = d + relativedelta(months=1)


if __name__ == '__main__':
    ride_duration_prediction_backfill()
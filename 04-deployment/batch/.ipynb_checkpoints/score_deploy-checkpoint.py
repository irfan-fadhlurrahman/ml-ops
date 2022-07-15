from prefect.deployments import DeploymentSpec
from prefect.orion.schemas.schedules import CronSchedule
from prefect.flow_runners import SubprocessFlowRunner

DeploymentSpec(
    flow_location="score.py",
    name="ride_duration_prediction",
    parameters={
        "taxi_type": "green",
        "run_id": "d3b600b0d89e4e95b27b5e333e4d3c01",
        
    },
    #flow_storage="b0f13ec1-588d-432c-b939-405f21ec4f69"
    schedule=CronSchedule(cron="0 3 2 * *"),
    flow_runner=SubprocessFlowRunner(),
    tags=["ml"]
)
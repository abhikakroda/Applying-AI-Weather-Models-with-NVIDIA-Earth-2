from datetime import datetime, timedelta

from earth2studio.data import GFS
from earth2studio.models.px.sfno import VARIABLES
from earth2studio.utils.time import to_time_array

if __name__ == "__main__":
    gfs = GFS()
    gfs(
        to_time_array([(datetime.now() - timedelta(hours=4)).isoformat()[:10]]),
        VARIABLES,
    )

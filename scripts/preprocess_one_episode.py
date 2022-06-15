from grit.core.data_processing import prepare_episode_dataset
from datetime import datetime
start = datetime.now()
prepare_episode_dataset(('bendplatz', 0, True))
print(datetime.now() - start)

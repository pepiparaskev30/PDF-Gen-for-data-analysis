'''import os
import pandas as pd
import matplotlib.pyplot as plt 

os.chdir("/Users/pepiparaskevoulakou/Desktop/Melodic_1/dataset_repo")

df_analysis  = pd.read_csv("09_02_2021.csv") 

print(df_analysis.head(10))
df_analysis.reset_index()
del(df_analysis['Datetime.1'])
#print(df_analysis.head(10))


plt.rc('font', size=12)
fig, ax = plt.subplots(figsize=(10, 6))

# Specify how our lines should look
ax.plot(df_analysis.Datetime, df_analysis.WillFinishTooSoonContext, color='tab:blue')

# Same as above
ax.set_xlabel('Time')
ax.set_ylabel('ETPercentile')
ax.set_title('Timeseries')
ax.grid(True)
ax.legend(loc='upper left');'''



''''import datetime

x1 = datetime.datetime.strptime(x1, '%Y-%m-%dT%H:%M:%SZ')'''




import datetime 
x2 = '2021-02-10T06:51:13.086Z'

x2 = datetime.datetime.strptime(x2, "%Y-%m-%dT%H:%M:%S.%fZ").timestamp()
import os
import pandas as pd
import numpy as np
import datetime

from dataset_morphemic import get_all_files

os.chdir("/Users/pepiparaskevoulakou/Desktop/Melodic_1/time-series-data/time-series-data/genome/deployment-reconfiguration-range-1-to-10/2021-02-18 to 2021-02-18/v1.0-raw data")
dataset_folder = "."

files = ['./EstimatedRemainingTimeContext.csv', './SimulationLeftNumber.csv', './SimulationElapsedTime.csv', 
    './NotFinishedOnTime.csv', './MinimumCoresContext.csv', './NotFinished.csv', './WillFinishTooSoonContext.csv', 
    './NotFinishedOnTimeContext.csv', './MinimumCores.csv', './ETPercentile.csv', './RemainingSimulationTimeMetric.csv', 
    './TotalCores.csv']


def get_all_files():
    _files = []
    for root, dirs, files in os.walk(dataset_folder):
        for filename in files:
            if not filename.endswith(".DS_Store"):
                _files.append(root + '/' + filename)
                features.append(filename[:-4])
    print(_files)
    #print(features)
    return _files, features 







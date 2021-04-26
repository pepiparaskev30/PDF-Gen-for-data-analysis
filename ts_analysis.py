
#import plotly.io as pio
#pio.renderers
'''import the necessary libraries'''
import pandas as pd # for using pandas daraframe
import numpy as np # for som math operations
from sklearn.preprocessing import StandardScaler # for standardizing the Data
from sklearn.decomposition import PCA # for PCA calculation
import matplotlib.pyplot as plt # for plotting
import seaborn as sns
from matplotlib import pyplot as plt
sns.set() 
import os
from scipy import stats #for calculating Z-score
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
from scipy.stats import shapiro
from scipy import stats
from statsmodels.graphics.gofplots import qqplot
import pylab 
import scipy.stats as stats
import warnings
warnings.filterwarnings("ignore")
from datetime import datetime
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np 
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw

os.chdir("/Users/pepiparaskevoulakou/Desktop/Melodic_1/dataset_repo")
file = "new_10_02_2021_range_4_to_4.csv"

class df_manipulation():
    def __init__(self, my_dataframe):
        self.my_dataframe = my_dataframe
        self.features = self.my_dataframe.columns
    
    def print_features(self):
        print(self.features)

    def redudant_features_removal(self):
        new_df = self.my_dataframe.filter(["Datetime",'SimulationLeftNumber','RemainingSimulationTimeMetric','ETPercentile',"SimulationElapsedTime"], axis=1)
        new_df = new_df.set_index('Datetime')
        return(new_df)
    
    def convert_unix_to_Datetime(df):
        df['Datetime'] = df.index
        df.index = np.arange(1, len(df) + 1)
        df = df[['Datetime','SimulationLeftNumber','RemainingSimulationTimeMetric','ETPercentile',"SimulationElapsedTime"]]
        df['Datetime'] = pd.to_datetime(df['Datetime'],unit='s')
        new_df = df.dropna()
        return(new_df)
    

'''create pairplot for data distribution and to trace the relationships between features'''
sns.pairplot(df_f)


'''# Visual Normality Checks - Histogram Plot
 Uniform and  left-skewed distributions  '''

fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(nrows=2, ncols=2,figsize=(15,10))
ax0.hist(df_f['SimulationLeftNumber'])
ax0.set_title('SimulationLeftNumber data distribution', fontsize = 13)
ax1.hist(df_f['RemainingSimulationTimeMetric'], color = 'r')
ax1.set_title('RemainingSimulationTimeMetric data distribution', fontsize = 13)
ax2.hist(df_f['ETPercentile'], color = 'g')
ax2.set_title('ETPercentile data distribution', fontsize = 13)
ax3.hist(df_f['SimulationElapsedTime'], color = 'y')
ax3.set_title('SimulationElapsedTime data distribution', fontsize = 13)
plt.show()

'''Quantile-Quantile Plot'''

qqplot(df_f['SimulationLeftNumber'], line='s')
qqplot(df_f['RemainingSimulationTimeMetric'], line='s')
qqplot(df_f['ETPercentile'], line='s')
qqplot(df_f['SimulationElapsedTime'], line='s')

'''Statistical Normality Tests'''

def shapiro_wilk(df):
    for i in df.columns:
        stat, p = shapiro(df[i])
        print('Statistics=%.3f, p=%.3f' % (stat, p))
        # interpret
        alpha = 0.05
        if p > alpha:
            print('Sample looks Gaussian (fail to reject H0)')
        else:
            print('Sample does not look Gaussian (reject H0)')
    return

def correlation_heatmap(df):
    plt.figure(figsize=(15, 8))
    plt.title("Correlation heatmap using spearman correlation method",fontsize=20)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    corr_df = df.corr(method='spearman')
    return(sns.heatmap(corr_df,
           annot=True,
           linewidths=0.4,
           annot_kws={'size': 14},cmap = "Blues"))

'''dynamic time warping for time-series features'''

def dtw(df):
    columns_ =[]
    row={}
    df_ = pd.DataFrame()
    for c in df.columns:
        if c!="Datetime":
            columns_.append(c)
    for i in columns_:
        for j in columns_:
            if j != i :
                dtw_distance, warp_path = fastdtw(df[i].values, df[j].values, dist=euclidean)
                row.update({"dtw": int(dtw_distance), "feature_1": i, "feature_2":j})
                df_ = df_.append(row, ignore_index=True)
                #print("DTW similarity for {0} and {1} is: ".format(i,j), int(dtw_distance))
    df_ = df_.drop_duplicates(subset=['dtw'])
    return(df_)

 def boxplot(df):
    df.index = np.arange(1, len(df) + 1)
    fig, ax = plt.subplots(figsize=(12,10))
    return(df.boxplot(['SimulationLeftNumber', 'RemainingSimulationTimeMetric',
    'ETPercentile', 'SimulationElapsedTime']))

def PCA_feature_importance(df):
    df = df.set_index('Datetime')
    X = df.values # getting all values as a matrix of dataframe 
    sc = StandardScaler() # creating a StandardScaler object
    X_std = sc.fit_transform(X) # standardizing the data
    pca = PCA(0.99) #create the amount of PCA that retains the 99% of the dataset's variance
    X_pca = pca.fit(X_std)
    fig, ax = plt.subplots(figsize=(10,9))
    plt.xlim(0,5)
    plt.plot(np.cumsum(pca.explained_variance_ratio_),color='tab:blue' )
    plt.xlabel('number of components')
    plt.ylabel('cumulative explained variance')
    related_df = pd.DataFrame(pca.components_, columns = df.columns)
    n_pcs= pca.n_components_ # get number of component
    # get the index of the most important feature on EACH component
    most_important = [np.abs(pca.components_[i]).argmax() for i in range(n_pcs)]
    initial_feature_names = df.columns
    # get the most important feature names
    most_important_names = [initial_feature_names[most_important[i]] for i in range(n_pcs)]
    return("Number of calculated PCA Components are:",pca.n_components_, "The most important features are:", most_important_names)

my_df = pd.read_csv(file, error_bad_lines=True)

a = df_manipulation(my_df)
s = a.redudant_features_removal()
s_ = a.convert_unix_to_Datetime()





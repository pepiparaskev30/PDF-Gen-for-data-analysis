import os
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from scipy.stats import shapiro
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
from sklearn import preprocessing
from matplotlib.backends.backend_pdf import PdfPages
import dataframe_image as dfi
from sklearn.preprocessing import StandardScaler # for standardizing the Data
from sklearn.decomposition import PCA
import dataframe_image as dfi

def find_the_file(path):
    for p_ in os.listdir(path):
        if p_.endswith(".csv"):
            file_ = p_
    return(file_)

def create_dataframe(file_):
    my_dataframe = pd.read_csv(file_, error_bad_lines=True)
    del(my_dataframe['Datetime.1'])
    return my_dataframe

class df_manipulation():
    def __init__(self, my_dataframe):
        self.my_dataframe = my_dataframe
        self.features = self.my_dataframe.columns
        
    def print_df(self):
        print(self.my_dataframe)
        return self.my_dataframe

    def print_features(self):
        print(self.features)

    def redudant_features_removal(self):
        self.new_df = self.my_dataframe.filter(["Datetime",'SimulationLeftNumber','RemainingSimulationTimeMetric','ETPercentile',"SimulationElapsedTime"], axis=1)
        self.new_df = self.new_df.set_index('Datetime')
        print(self.new_df)
        return(self.new_df)

    #Dataframe clean with Datetime as a column
    def convert_unix_to_Datetime(self):
        self.my_dataframe['Datetime'] = pd.to_datetime(self.my_dataframe['Datetime'], unit="s")
        #self.my_dataframe['Datetime'] = self.my_dataframe.index
        self.my_dataframe.index = np.arange(1, len(self.my_dataframe) + 1)
        self.my_dataframe = self.my_dataframe[['Datetime','SimulationLeftNumber','RemainingSimulationTimeMetric','ETPercentile',"SimulationElapsedTime"]]
        self.my_dataframe = self.my_dataframe.dropna()
        #print(self.my_dataframe)
        return(self.my_dataframe)
    
    #Dataframe clean with Datetime as an index
    def set_index_datetime(self):
        self.new_df = self.convert_unix_to_Datetime()
        self.new_df.set_index('Datetime', inplace=True)
        #print(self.new_df)
        return(self.new_df)
    
    #Dataframe clean with normal index without Datetime
    def clear_df(self):
        self.cleardf = self.convert_unix_to_Datetime()
        self.cleardf.index = np.arange(1, len(self.cleardf) + 1)
        self.cleardf = self.cleardf.drop(['Datetime'], axis=1)
        #print(self.cleardf)
        return(self.cleardf)

    #Dataframe with normalized values
    def normalized(self):
        x = self.clear_df().values #returns a numpy array
        min_max_scaler = preprocessing.MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(x)
        self.norm = pd.DataFrame(x_scaled, columns = ['SimulationLeftNumber','RemainingSimulationTimeMetric','ETPercentile',"SimulationElapsedTime"])
        #print(self.norm)
        return(self.norm)

    #Correlogram with the clear_df() dataset by using the spearman method
    def correlogram(self):
        self.corr_df = self.clear_df().corr(method='spearman')
        #plt.title("Correlation heatmap using spearman correlation method",fontsize=20)
        #plt.xticks(rotation=90);
        #plt.yticks(rotation=0);
        fig, ax = plt.subplots(figsize=(14,10))
        # Customize the heatmap of the corr_meat correlation matrix
        sns.heatmap(self.corr_df,
                annot=True,
                linewidths=0.4,
                annot_kws={'size': 14},cmap = "Blues")
        #fig_ = plt.show()
        return fig
    
    def ts_visual(self):
        df_an = self.set_index_datetime().reset_index()
        plot = df_an.plot(x="Datetime")
        fig = plot.get_figure()
        return fig

    def pairplot(self):
        g = sns.pairplot(self.clear_df())
        g.fig.suptitle("Your plot title") # y= some height>1        
        return g.fig
        
    def boxplot(self):
        fig, ax = plt.subplots(figsize=(12,10))
        self.clear_df().boxplot(['SimulationLeftNumber', 'RemainingSimulationTimeMetric',
                        'ETPercentile', 'SimulationElapsedTime'])
        #fig_ = plt.show()
        return fig

    def hist(self):
        fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(nrows=2, ncols=2,figsize=(15,10))
        ax0.hist(self.clear_df()['SimulationLeftNumber'])
        ax0.set_title('SimulationLeftNumber data distribution', fontsize = 13)
        ax1.hist(self.clear_df()['RemainingSimulationTimeMetric'], color = 'r')
        ax1.set_title('RemainingSimulationTimeMetric data distribution', fontsize = 13)
        ax2.hist(self.clear_df()['ETPercentile'], color = 'g')
        ax2.set_title('ETPercentile data distribution', fontsize = 13)
        ax3.hist(self.clear_df()['SimulationElapsedTime'], color = 'y')
        ax3.set_title('SimulationElapsedTime data distribution', fontsize = 13)
        return(fig)
    
    def Shapiro_wilk_test(self):
        results=[]
        for i in self.clear_df().columns:
            stat, p = shapiro(self.clear_df()[i])
            #print('Statistics=%.3f, p=%.3f' % (stat, p))
            # interpret
            alpha = 0.05
            if p > alpha:
                results.append('For the feature: {} Sample looks Gaussian (fail to reject H0)'.format(i))
            else:
                results.append('For the feature {}: Sample does not look Gaussian (reject H0)'.format(i))
        results = '\n'.join(results)
        return results

    def Dynamic_Time_Warping(self):
        columns_ =[]
        row={}
        df_ = pd.DataFrame()
        for c in self.normalized().columns:
            columns_.append(c)
        for i in columns_:
            for j in columns_:
                if j != i :
                    dtw_distance, warp_path = fastdtw(self.normalized()[i].values, self.normalized()[j].values, dist=euclidean)
                    row.update({"dtw": int(dtw_distance), "feature_1": i, "feature_2":j})
                    df_ = df_.append(row, ignore_index=True)
                    #print("DTW similarity for {0} and {1} is: ".format(i,j), int(dtw_distance))
        df_= df_.drop_duplicates(subset=['dtw'])
        #print(df_)
        return df_
    
    #Principal component analysis for feature extraction
    def PCA_feature_importance(self):
        df = self.clear_df()
        X = df.values # getting all values as a matrix of dataframe 
        sc = StandardScaler() # creating a StandardScaler object
        X_std = sc.fit_transform(X) # standardizing the data
        pca = PCA(0.99) #create the amount of PCA that retains the 99% of the dataset's variance
        X_pca = pca.fit(X_std)
        fig, ax = plt.subplots(figsize=(10,10))
        plt.xlim(0,5)
        plt.plot(np.cumsum(pca.explained_variance_ratio_),color='tab:red' )
        plt.xlabel('number of components')
        plt.ylabel('cumulative explained variance')
        related_df = pd.DataFrame(pca.components_, columns = df.columns)
        n_pcs= pca.n_components_ # get number of component
        # get the index of the most important feature on EACH component
        most_important = [np.abs(pca.components_[i]).argmax() for i in range(n_pcs)]
        initial_feature_names = df.columns
        # get the most important feature names
        most_important_names = [initial_feature_names[most_important[i]] for i in range(n_pcs)]
        result = "The most important initial features are:", most_important_names[0], most_important_names[1]
        plt.title(result)
        #plt.show()
        return fig

#function in order to depict a dataframe as an img

def render_mpl_table(data, col_width=3.0, row_height=0.625, font_size=6,
                     header_color='#40466e', row_colors=['#f1f1f2', 'w'], edge_color='w',
                     bbox=[0, 0, 1, 1], header_columns=0, 
                     ax=None, **kwargs):
    if ax is None:
        size = (np.array(data.shape[::-1]) + np.array([0, 1])) * np.array([col_width, row_height])
        fig, ax = plt.subplots(figsize=size)
        ax.axis('off')
    mpl_table = ax.table(cellText=data.values, bbox=bbox, colLabels=data.columns, **kwargs)
    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(font_size)

    for k, cell in mpl_table._cells.items():
        cell.set_edgecolor(edge_color)
        if k[0] == 0 or k[1] < header_columns:
            cell.set_text_props(weight='bold', color='w')
            cell.set_facecolor(header_color)
        else:
            cell.set_facecolor(row_colors[k[0]%len(row_colors) ])
    return ax.get_figure(), ax

#fig,ax = render_mpl_table(df_.head(10), header_columns=0, col_width=2.0)






   






    
    



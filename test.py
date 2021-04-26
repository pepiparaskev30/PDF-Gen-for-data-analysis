import os
from time_series import find_the_file, create_dataframe, render_mpl_table
from time_series import df_manipulation
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image, ImageDraw
from matplotlib.pyplot import figure
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
import seaborn as sns
from datetime import datetime

#Generate the proper dataset format from the class df_manipulation that belngs to the time_series.py
def generate_the_dataset(file_):
    df = create_dataframe(file_)
    df_ = df_manipulation(df)
    return(df_)
#Function for generating the time that the pdf will be created
def generate_time():
    now = datetime.now()
    return(now.strftime("%d/%m/%Y %H:%M:%S"))
#Function for creating the proper plots
def generate_figs(plt_obj, txt:str):
    fig = plt_obj
    fig.suptitle(txt, fontsize=15)
    fig.set_size_inches(16, 12)
    return fig 
    



#Actions to be taken for generating the pdf report 
if __name__ == "__main__":
    path = "/Users/pepiparaskevoulakou/Desktop/Projects/Melodic_1/src/results/saved/"
    file_ = find_the_file(path)
    os.chdir("./results/saved")
    df_ = generate_the_dataset(file_)
    pp = PdfPages("Data_Analysis1.pdf")
    firstPage = plt.figure(figsize=(11.69,8.27))
    txt = 'Report Analysis for dataset {}'.format(file_)
    txt_1 = 'Generated at: {}'.format(generate_time())
    firstPage.text(0.5,0.5,txt, transform=firstPage.transFigure, size=24, ha="center")
    firstPage.text(0.3,0.3,txt_1, transform=firstPage.transFigure, size=18, ha="center")
    pp.savefig(firstPage)
    fig_1, ax = render_mpl_table(df_.clear_df().head(10), header_columns=0, col_width=2.0)
    fig_1.suptitle("Dataset concerning the file {}".format(file_), fontsize=15)
    pp.savefig(fig_1)
    S_w_page = plt.figure(figsize=(11.69,8.27))
    S_w_page.text(0.1,0.8, "Statistical Normality test \n"+df_.Shapiro_wilk_test(), transform=S_w_page.transFigure, size=13)
    pp.savefig(S_w_page)
    fig_1 = generate_figs(df_.boxplot(), "Boxplot")
    pp.savefig(fig_1)
    fig_2 = generate_figs(df_.pairplot(), "Pairplot")
    pp.savefig(fig_2)
    fig_3 = generate_figs(df_.correlogram(), "Correlation heatmap")
    pp.savefig(fig_3)
    fig_4 = generate_figs(df_.hist(), "Histograms")
    pp.savefig(fig_4)
    fig_5 = generate_figs(df_.PCA_feature_importance(), "PCA visualization")
    pp.savefig(fig_5)
    fig_6, ax = render_mpl_table(df_.Dynamic_Time_Warping(), header_columns=0, col_width=2.0)
    fig_6.suptitle("Dynamic Time Warping (similarity method)",fontsize=15)
    pp.savefig(fig_6)
    df_ = generate_the_dataset(file_)
    fig_7 = generate_figs(df_.ts_visual(), "Multivariante ts")
    pp.savefig(fig_7)
    pp.close()


'''fig_2 = df_.boxplot()
fig_2.suptitle("Boxplot", fontsize=15)
fig_2.set_size_inches(16, 12)
pp.savefig(fig_2)
fig_3 = df_.pairplot()
fig_3.suptitle("Pairplot", fontsize = 15)
fig_3.set_size_inches(16, 12)
pp.savefig(fig_3)
fig_4 = df_.correlogram()
fig_4.suptitle("Correlation heatmap", fontsize=15)
fig_4.set_size_inches(16, 12)
pp.savefig(fig_4)
fig_5 = df_.hist()
fig_5.suptitle("Histograms", fontsize=15)
pp.savefig(fig_5)
fig_6 = df_.PCA_feature_importance()
fig_6.suptitle("PCA visualization", fontsize=15)
fig_6.set_size_inches(16, 12)
pp.savefig(fig_6)
fig_8, ax = render_mpl_table(df_.Dynamic_Time_Warping(), header_columns=0, col_width=2.0)
fig_8.suptitle("Dynamic Time Warping (similarity method)",fontsize=15)
pp.savefig(fig_8)
pp.close()'''



    



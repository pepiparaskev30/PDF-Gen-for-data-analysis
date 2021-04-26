from flask import Flask, render_template, request, redirect, url_for, Response, session, redirect
import io
import random
import os
from os.path import join, dirname, realpath
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy import stats
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
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from flask import Flask, render_template, request, redirect, url_for
import os
from os.path import join, dirname, realpath
from time_series import df_manipulation
from time_series import create_dataframe, find_the_file
import json
from scipy.stats import shapiro

app = Flask(__name__)

# enable debugging mode
app.config["DEBUG"] = True

#path to save the files
path_for_saving = "results/saved"

# Upload folder
UPLOAD_FOLDER = '/Users/pepiparaskevoulakou/Desktop/Projects/Melodic_1/dataset_repo'
app.config['UPLOAD_FOLDER'] =  UPLOAD_FOLDER

# Root URL
@app.route('/')
def index():
     # Set The upload HTML template '\templates\index.html'
    return render_template('index.html')

# Get the uploaded files
@app.route("/", methods=['POST'])
def uploadFiles():
    # get the uploaded file
    uploaded_file = request.files['file']
    if uploaded_file.filename.endswith(".csv"):
        # set the file path
        file_path = os.path.join(os.path.join(os.getcwd(),path_for_saving),uploaded_file.filename )
        # save the file
        uploaded_file.save(file_path) 
        result = "Success"
    elif not uploaded_file.filename.endswith(".csv"):
        result = "Please provide a valid file"
    else:
        result = "Unable to fetch the data"
    return render_template('submit.html', output = result)


@app.route("/download",methods=['GET'])
def download_full_report():
    path = os.path.join(os.getcwd(), "results/saved")
    file_ = find_the_file(path)
    df = create_dataframe(find_the_file)
    #final_df = df_manipulation(df)
    return render_template('simple.html',  tables=[df.to_html(classes='data')], titles=df.columns.values)


if (__name__ == "__main__"):
     app.run(port = 5001)
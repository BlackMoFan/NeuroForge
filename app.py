from flask import Flask, render_template, request, send_file
import pandas as pd
from collections import Counter
import numpy as np
import random as rd
import os
import matplotlib.pyplot as plt
from keras.models import load_model
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import train_test_split
import nltk
import re


app = Flask(__name__)

max_length = 150
trunc_type = 'post'
padding_type = 'post'

@app.route('/')
def index():
    return render_template('index.html')

#--------------- KNN PART
def euclidean_distance(x1, x2):
    distance = np.sqrt(np.sum((x1-x2)**2))
    return distance

class KNN:
    def __init__(self, k=3):
        self.k = k

    def Fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def Predict(self, X):
        predictions = [self._Predict(x) for x in X]
        return predictions

    def _Predict(self, x):
        # compute the distance
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
    
        # get the closest k
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        # majority vote
        most_common = Counter(k_nearest_labels).most_common()
        return most_common[0][0]

@app.route('/KNNPage')
def activity1KNN():
    return render_template('activity1knn.html')

@app.route('/get_inputknn', methods=['POST'])
def get_inputknn():
    #data from form
    knnexperience = int(request.form['knnexperience'])
    knnsalary = int(request.form['knnsalary'])
    #processing of data
    data = pd.read_csv('tokenizers/Jobstreet.csv')
    X = data[['Experience', 'Salary']].values
    y = data['Job'].values
    # Euclidean Distance with scikitlearn
    # from sklearn.neighbors import KNeighborsClassifier
    # from sklearn.model_selection import train_test_split
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)
    # knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')# (n_neighbors=5) original
    # knn.fit(X_train, y_train)
    # knnpredicted = knn.predict(np.array([knnexperience, knnsalary]))
    # Euclidean Distance Manually 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = KNN(k=2)#(k=5) original
    clf.Fit(X_train, y_train)
    knnpredicted = clf.Predict(np.array([knnexperience, knnsalary]))

    # html_list = "<ol>"
    # html_list = ""
    # for item in list(knnpredicted):
    #     html_list += "<li>{}</li><br />".format(item)
    # html_list += "</ol>"
    # return html_list

    return render_template('activity1knn.html', knnexperience=knnexperience, knnsalary=knnsalary, knnprediction=knnpredicted)

#--------------- KNN PART


#--------------- KMEANS PART
@app.route('/KMeansPage')
def activity2KMeans():
    return render_template('activity2kmeans.html')


@app.route('/get_inputkmeans', methods=['POST'])
def get_inputkmeans():
    import base64

    kmeansNumCluster = int(request.form['kmeansNumCluster'])
    
    # df = pd.read_csv("tokenizers/indeedscrape.csv")
    # df['Salary Rate']=df['Salary Rate'].str.replace(',','')
    #processing of dataimport pandas as pd
    data = pd.read_csv('tokenizers/Jobstreet.csv', encoding='utf-8')
    jobs = data['Job'].tolist()
    # print(jobs)
        
    # Create a dictionary to store the jobs and integers
    jobs_dictionary = {}

    # Iterate over the list of jobs
    for job in jobs:
        # Assign each job to an integer
        jobs_dictionary[jobs.index(job)] = job

    # Print the dictionary
    # print(jobs_dictionary)

    data['jobToInt'] = 0

    for i in range(len(data)):
        data['jobToInt'][i] = i
        
    # print(data)

    X = data.iloc[:, [4,3]].values # use Job and Salary
    # number of training samples
    m = X.shape[0]
    # number of features
    n = X.shape[1]
    # choosing the number of iteration which guarantee convergence
    n_iterations = 1000
    # number of clusters
    K = kmeansNumCluster
    # Step 1. Initialize the centroids randomly
    centroids = np.array([]).reshape(n,0)
    for i in range(kmeansNumCluster):
        rand = rd.randint(0, m-1) # random
        centroids = np.c_[centroids, X[rand]] # randomize centroids

    # Main K-Means Clustering Part
    from collections import defaultdict
    Output = defaultdict()
    # output = {}

    for i in range(n_iterations):
        #step 2a.
        lst1 = [] # create list
        lst = np.array(lst1) # convert list to numpy array
        ED = np.array(lst).reshape(m, 0)
        for k in range(K):
            temporary_distance = np.sum((X-centroids[:, k])**2, axis=1)
            ED = np.c_[ED, temporary_distance]
        
        C = np.argmin(ED, axis=1)+1
        #step 2b.
        
        Y = {}
        lst1 = [] # create list
        lst = np.array(lst1) # convert list to numpy array
        for k in range(K):
            Y[k+1] = np.array(lst.reshape(2,0))
            #Horizontal Concatenation, regrouping -> clustered index C
        for i in range(m):
            Y[C[i]] = np.c_[Y[C[i]], X[i]]
            #Transpose
        for k in range(K):
            Y[k+1] = Y[k+1].T
            #Mean Computation & New Assigned centroid
        for k in range(K):
            centroids[:, k] = np.mean(Y[k+1], axis=0)

        Output = Y
        # print("Output:", output)

    color = ['red', 'blue', 'green', 'cyan', 'magenta', 'grey', 'yellow', 'pink', 'brown', 'orange']
    labels = ['cluster#1', 'cluster#2', 'cluster#3', 'cluster#4', 'cluster#5', 'cluster#6', 'cluster#7', 'cluster#8', 'cluster#9', 'cluster#10']

    for k in range(K):
        plt.scatter(Output[k+1][:,0],
            Output[k+1][:,1],
            c=color[k],
            label=labels[k])
        
    plt.scatter(centroids[0,:], centroids[1,:], s=150, c='yellow', label='centroid')
    plt.xlabel('Job')
    plt.ylabel('Salary')
    # plt.legend()
    plt.title('Plot of data points')

    # scaler = MinMaxScaler()

    # scaler.fit(df[['Salary Rate']])
    # df['Salary Rate'] = scaler.transform(df[['Salary Rate']])

    # scaler.fit(df[['Experience']])
    # df['Experience'] = scaler.transform(df[['Experience']])

    # km = KMeans(n_clusters=kmeansNumCluster)
    # y_predicted = km.fit_predict(df[['Experience','Salary Rate']])
    # y_predicted
    # df['cluster']=y_predicted 
    # km.cluster_centers_

    # km = KMeans(n_clusters=kmeansNumCluster)
    # y_predicted = km.fit_predict(df[['Experience','Salary Rate']])
    # y_predicted
    # df['cluster']=y_predicted
    # km.cluster_centers_
    # df1 = df[df.cluster==0]
    # df2 = df[df.cluster==1]
    # df3 = df[df.cluster==2]
    # df4 = df[df.cluster==3]
    # df5 = df[df.cluster==4]
    # plt.scatter(df1.Experience,df1['Salary Rate'],color='green')
    # plt.scatter(df2.Experience,df2['Salary Rate'],color='red')
    # plt.scatter(df3.Experience,df3['Salary Rate'],color='black')
    # plt.scatter(df4.Experience,df4['Salary Rate'],color='Blue')
    # plt.scatter(df5.Experience,df5['Salary Rate'],color='Yellow')
    # plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],color='purple',marker='*',label='centroid')
    # plt.legend()
    # plt.xlabel('Experience')
    # plt.ylabel('Salary Rate')

    path = 'static/images/plot/kmeans_plot.png'
    if os.path.exists(path):
        os.remove(path)
    plt.savefig(path, format='png')

    with open(path, 'rb') as img_file:
        encoded_img = base64.b64encode(img_file.read()).decode('utf-8')
    plt.close()
    # plt.show()
    # Save the figure in the static directory 
    # plt.savefig(os.path.join('static', 'images', 'plot', 'plot.png'))
    # plt.close()
    # return plt
    return render_template('activity2kmeans.html', kmeansimg=encoded_img)

#--------------- KMEANS PART

#--------------- NBayes PART
@app.route('/NBayesPage')
def activity3NBayes():
    return render_template('activity3nbayes.html')

@app.route('/get_inputNB', methods=["POST", "GET"])
def get_inputNB():
    import base64
    df = pd.read_csv('tokenizers/indeedscrape.csv')

    #Pre-process
    df['Salary Rate']=df['Salary Rate'].str.replace(',','')

    def convert_to_int(word):
        word_dict = {'Junior web programmer': 1, 'Junior IT Programmer': 1, 'Junior Programmer': 1, 'Junior programmer': 1, 'Senior Programmer': 3, 'senior Programmer': 3, 'Senior game Developer': 3, 'Senior IT System Analyst': 3,
                    'CICS Developer': 2,'Senior Analyst Programmer': 3,'Sr. Systems Analyst/Programmer': 3,'Senior Web Programmer': 3,'C# Developer': 2,'Senior SAP ABAP Programmer': 3,'Workday Support Programmer': 2,
                    'Computer Programmer': 2,'CNC Programmer': 2,'Senior Backend Developer': 3,'Angular Web Programmer-Junior': 1,'Senior ABAP Programmer': 3,'Senior Full Stack .NET Developer': 3,'Senior WordPress Developer': 3,
                    'Junior PHP Developer': 1, 'Senior PHP Developer': 3,'VBA EXCEL MACRO PROGRAMMER': 2,'IT Programmer': 2,'C# Programmer': 2,'Angular Web Programmer - Junior': 1,'NodeJs Developer': 2,'Web Developer': 2,
                    'Front End Developer': 2,'System Programmer': 2,'Analyst Programmer': 2,'Japanese Senior Web Programmer': 3,'Sr. Analyst Programmer': 3,'Sr. SAP ABAP Programmer': 3,'.Net Developer': 2,
                    'Senior .Net Developers': 3,'Senior React Analyst Programmer': 3,'COMPUTER PROGRAMMER': 2,'SAP ABAP Programmer': 2,'Web Developer (PHP)': 2,'Senior Web Developer': 3,'Senior Frontend/Fullstack Developer': 3,'PHP Developer': 2,
                    'Back End Developer': 2,'Senior Python Developer': 3,'Java Developer': 2,'Senior Mobile App Developer': 3,'ASP.Net Developer': 2,'FULL-TIME Senior WordPress Developer': 3,'Jr. Net and SQL Programmer': 1,
                    'Junior .Net Developer': 1,'Cobol/CICS Developer': 2,'Cobol Developer': 2,'Mid to Senior Full Stack Developer': 3,'Remote Python Developer': 2,'Sr. C#.Net Developer': 3,'PYTHON DEVELOPER': 2,
                    'C#.NET Developer': 2,'Junior Web Developer': 1,'Senior Full Stack .NET Developer (Work From Home)': 3, 'Senior Wordpress Developer - (Remote)': 3, 'Web Developer (NodeJS)': 2,
                    'SENIOR Software Developer to work from Makati': 3, 'Senior Front-End Developer - Work From Home': 3, 'Senior Outsystems Developer': 3,  'Application Developers (Mid/Sr) - C#, Java, Cobol': 3,
                    'Senior Web Developer REACT': 3, 'Senior Golang Developer': 3, 'Jr. Developer / Analyst': 1, 'Junior Database Developer': 1, 'Work From Home - Web Developer': 2, 'Full-Stack Developer': 2,
                    'Senior Frontend Developer': 3,  'Senior Backend Developer / PHP / Remote (m/f/d)': 3, 'Full Stack Developer': 2,  'Senior Database Developer': 2, 'Senior React Developer (Remote)': 3,
                    'Senior VB.NET Winform Desktop Application Developer': 3, 'Node. Js Developer': 1, 'ReactJS Developer': 2, 'Senior Game Developer': 3, 'Jr. Programmer': 1, 'Programmer': 2, 'Senior UI Developer': 3,
                    'Back End Developer Senior': 3,
                    }
        if word is not None and word in word_dict:
            return word_dict[word]
        else:
            return None
        
    def convert_to_word(num):
        word_dict = {'Junior web programmer': 1, 'Junior IT Programmer': 1, 'Junior Programmer': 1, 'Junior programmer': 1, 'Senior Programmer': 3, 'senior Programmer': 3, 'Senior game Developer': 3, 'Senior IT System Analyst': 3,
                    'CICS Developer': 2,'Senior Analyst Programmer': 3,'Sr. Systems Analyst/Programmer': 3,'Senior Web Programmer': 3,'C# Developer': 2,'Senior SAP ABAP Programmer': 3,'Workday Support Programmer': 2,
                    'Computer Programmer': 2,'CNC Programmer': 2,'Senior Backend Developer': 3,'Angular Web Programmer-Junior': 1,'Senior ABAP Programmer': 3,'Senior Full Stack .NET Developer': 3,'Senior WordPress Developer': 3,
                    'Junior PHP Developer': 1, 'Senior PHP Developer': 3,'VBA EXCEL MACRO PROGRAMMER': 2,'IT Programmer': 2,'C# Programmer': 2,'Angular Web Programmer - Junior': 1,'NodeJs Developer': 2,'Web Developer': 2,
                    'Front End Developer': 2,'System Programmer': 2,'Analyst Programmer': 2,'Japanese Senior Web Programmer': 3,'Sr. Analyst Programmer': 3,'Sr. SAP ABAP Programmer': 3,'.Net Developer': 2,
                    'Senior .Net Developers': 3,'Senior React Analyst Programmer': 3,'COMPUTER PROGRAMMER': 2,'SAP ABAP Programmer': 2,'Web Developer (PHP)': 2,'Senior Web Developer': 3,'Senior Frontend/Fullstack Developer': 3,'PHP Developer': 2,
                    'Back End Developer': 2,'Senior Python Developer': 3,'Java Developer': 2,'Senior Mobile App Developer': 3,'ASP.Net Developer': 2,'FULL-TIME Senior WordPress Developer': 3,'Jr. Net and SQL Programmer': 1,
                    'Junior .Net Developer': 1,'Cobol/CICS Developer': 2,'Cobol Developer': 2,'Mid to Senior Full Stack Developer': 3,'Remote Python Developer': 2,'Sr. C#.Net Developer': 3,'PYTHON DEVELOPER': 2,
                    'C#.NET Developer': 2,'Junior Web Developer': 1,'Senior Full Stack .NET Developer (Work From Home)': 3, 'Senior Wordpress Developer - (Remote)': 3, 'Web Developer (NodeJS)': 2,
                    'SENIOR Software Developer to work from Makati': 3, 'Senior Front-End Developer - Work From Home': 3, 'Senior Outsystems Developer': 3,  'Application Developers (Mid/Sr) - C#, Java, Cobol': 3,
                    'Senior Web Developer REACT': 3, 'Senior Golang Developer': 3, 'Jr. Developer / Analyst': 1, 'Junior Database Developer': 1, 'Work From Home - Web Developer': 2, 'Full-Stack Developer': 2,
                    'Senior Frontend Developer': 3,  'Senior Backend Developer / PHP / Remote (m/f/d)': 3, 'Full Stack Developer': 2,  'Senior Database Developer': 2, 'Senior React Developer (Remote)': 3,
                    'Senior VB.NET Winform Desktop Application Developer': 3, 'Node. Js Developer': 1, 'ReactJS Developer': 2, 'Senior Game Developer': 3, 'Jr. Programmer': 1, 'Programmer': 2, 'Senior UI Developer': 3,
                    'Back End Developer Senior': 3,
                    }
        for key, val in word_dict.items():
            if num == val:
                return key
            else:
                return "None"
        
    # df['Job'] = df['Job'].apply(lambda x : convert_to_int(x))

    X = df[['Experience', 'Salary Rate']].values
    y = df['Job'].values

    NBOption = int(request.form['NBOption'])
    nbExp = int(request.form["nbExp"])
    nbSalary = int(request.form["nbSalary"])

    if NBOption == 1:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=50)
        model= GaussianNB()
        model.fit(X_train,y_train)
        model_name = type(model).__name__  


        test_sizes = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1] 
        accuracies = []

        nbPredicted = model.predict(np.array([[nbExp, nbSalary]]))
        # print("nbPredicted =", nbPredicted[0], ", type = ", type(int(nbPredicted[0])))
        # nbPredictedWord = convert_to_word(nbPredicted)

        for size in test_sizes:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=size, random_state=42)
            clf = GaussianNB()
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            accuracy = clf.score(X_test, y_test)
            accuracies.append(accuracy)


        plt.plot(test_sizes, accuracies, marker='o')
        plt.xlabel('Test Size')
        plt.ylabel('Accuracy')
        plt.title('Accuracy vs. Test Size based on ' + model_name)

        plot_pathnb = 'static/nb_plot.png'
        if os.path.exists(plot_pathnb):
            os.remove(plot_pathnb)
        plt.savefig(plot_pathnb, format='png')


        with open(plot_pathnb, 'rb') as img_file:
            encoded_imgnb = base64.b64encode(img_file.read()).decode('utf-8')
        plt.close()
        return render_template('activity3nbayes.html', nbimg=encoded_imgnb, nbPredictedWord=nbPredicted, nbExp=nbExp, nbSalary=nbSalary)

    elif NBOption == 2:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=50)
        model= BernoulliNB()
        model.fit(X_train,y_train)
        model_name = type(model).__name__  


        test_sizes = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1] 
        accuracies = []

        # model= GaussianNB()
        # model.fit(X_train,y_train)
        nbPredicted = model.predict(np.array([[nbExp, nbSalary]]))
        # nbPredictedWord = convert_to_word(nbPredicted)

        for size in test_sizes:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=size, random_state=42)
            clf = BernoulliNB()
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            accuracy = clf.score(X_test, y_test)
            accuracies.append(accuracy)


        plt.plot(test_sizes, accuracies, marker='o')
        plt.xlabel('Test Size')
        plt.ylabel('Accuracy')
        plt.title('Accuracy vs. Test Size based on ' + model_name)

        plot_pathnb = 'static/nb_plot.png'
        if os.path.exists(plot_pathnb):
            os.remove(plot_pathnb)
        plt.savefig(plot_pathnb, format='png')


        with open(plot_pathnb, 'rb') as img_file:
            encoded_imgnb = base64.b64encode(img_file.read()).decode('utf-8')
        plt.close()
        return render_template('activity3nbayes.html', nbimg=encoded_imgnb, nbPredictedWord=nbPredicted, nbExp=nbExp, nbSalary=nbSalary)
    
    elif NBOption == 3:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=50)
        model= MultinomialNB()
        model.fit(X_train,y_train)
        model_name = type(model).__name__  


        test_sizes = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1] 
        accuracies = []

        # model= GaussianNB()
        # model.fit(X_train,y_train)
        nbPredicted = model.predict(np.array([[nbExp, nbSalary]]))
        # nbPredictedWord = convert_to_word(nbPredicted)

        for size in test_sizes:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=size, random_state=42)
            clf = MultinomialNB()
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            accuracy = clf.score(X_test, y_test)
            accuracies.append(accuracy)


        plt.plot(test_sizes, accuracies, marker='o')
        plt.xlabel('Test Size')
        plt.ylabel('Accuracy')
        plt.title('Accuracy vs. Test Size based on ' + model_name)

        plot_pathnb = 'static/nb_plot.png'
        if os.path.exists(plot_pathnb):
            os.remove(plot_pathnb)
        plt.savefig(plot_pathnb, format='png')


        with open(plot_pathnb, 'rb') as img_file:
            encoded_imgnb = base64.b64encode(img_file.read()).decode('utf-8')
        plt.close()
        return render_template('activity3nbayes.html', nbimg=encoded_imgnb, nbPredictedWord=nbPredicted, nbExp=nbExp, nbSalary=nbSalary)
    else: 
        plt.close()
        return render_template('activity3nbayes.html')

#--------------- NBayes PART

#--------------- Regression PART
@app.route('/RegressionPage')
def activity4Regression():
    return render_template('activity4regression.html')

@app.route('/get_inputReg', methods=["POST", "GET"])
def get_inputReg():
    regModel = pickle.load(open('models/regModel.pkl','rb'))

    regJob = float(request.form["regJob"])
    regExperience = float(request.form["regExperience"])

    regPrediction = regModel.predict([[regJob, regExperience]])

    return render_template('activity4regression.html', regPrediction=regPrediction, regJob=regJob, regExperience=regExperience)

#--------------- Regression PART

#--------------- TextGen PART
@app.route('/TextGenPage')
def activity5TextGen():
    return render_template('activity5textgen.html')

@app.route('/get_inputTextGen', methods=["POST", "GET"])
def get_inputTextGen():
    loaded_model = ""
    genType = request.form["genType"]
    if genType == "GRU":
        loaded_model = load_model('models/GRUModelsT/model_20230608_190827.h5')
        tokenizer = pickle.load(open('tokenizers/GRU TokenizerT/tokenizer_20230608_190827.pkl', 'rb'))
    elif genType == "LSTM":
        loaded_model = load_model('models/LSTMModelsT/model_20230608_185915.h5')
        tokenizer = pickle.load(open('tokenizers/LSTM TokenizerT/tokenizer_20230608_190043.pkl', 'rb'))
    elif genType == "Bi-LSTM":
        loaded_model = load_model('models/Bi-LSTMModelsT/model_20230608_192510.h5')
        tokenizer = pickle.load(open('tokenizers/Bi-LSTM TokenizerT/tokenizer_20230608_192510.pkl', 'rb'))
    
    num_words = int(request.form["num_words"])

    data = open('tokenizers/irish-lyrics-eof.txt').read()
    # corpus data normalized -> lower, split into strings by new line characters
    c_data = data.lower().split("\n")

    input_sequences = []

    for line in c_data:
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[:i+1]
            input_sequences.append(n_gram_sequence)

    # pad sequences
    max_sequence_len = max([len(x) for x in input_sequences])

    seed_text = request.form["sentence"]

    def generate_text(seed_text, num_words):
        for _ in range(num_words):
            token_list = tokenizer.texts_to_sequences([seed_text])[0]
            token_list = pad_sequences([token_list],maxlen=max_sequence_len-1,padding='pre')
            predicted = np.argmax(loaded_model.predict(token_list), axis=-1)
            output_word = ""
            for word, index in tokenizer.word_index.items():
                if index == predicted:
                    output_word = word
                    break
            seed_text += " " + output_word

        return seed_text
    
    generated_text = generate_text(seed_text, num_words)

    return render_template('activity5textgen.html', textgenResult=generated_text, genType=genType, num_words=num_words)


#--------------- TextGen PART

#--------------- Classification PART 
@app.route('/ClassificationPage')
def activity6Classification():
    return render_template('activity6classification.html')

@app.route('/get_inputClassification', methods=["POST", "GET"])

def get_inputClassification():
    import tensorflow as tf
    
    # Check if the required fields are empty
    if "sentence" not in request.form or "genType" not in request.form:
        error_message = "Please enter a sentence and select a model type."
        return render_template('activity6classification.html', error_message=error_message)
    
    loaded_model = ""
    genType = request.form["genType"]
    if genType == "GRU":
        loaded_model = load_model('models/GRUModelsC/model_20230607_211248.h5')
        with open('tokenizers/GRU Tokenizer/tokenizer_20230607_211505.pkl', 'rb') as handle:
            tokenizer = pickle.load(handle)
    elif genType == "LSTM":
        loaded_model = load_model('models/LSTMModelsC/model_20230607_205656.h5')
        with open('tokenizers/LSTM Tokenizer/tokenizer_20230607_210039.pkl', 'rb') as handle:
            tokenizer = pickle.load(handle)
    elif genType == "Bi-LSTM":
        loaded_model = load_model('models/Bi-LSTMModelsC/model_20230607_225937.h5')
        with open('tokenizers/Bi-LSTM Tokenizer/tokenizer_20230607_225937.pkl', 'rb') as handle:
            tokenizer = pickle.load(handle)
    
    model = loaded_model
    # sentence = ['Did this man apologise on set for a small joke and had the audience applaud the guy doing his job. This guy is the nicest person Ive seen.',
    #         'I am ugly af']

    sentence = request.form["sentence"]
    sentence_output = str(sentence)
    
    # Hyperparameters of the model
    vocab_size = 3000 # choose based on statistics
    oov_tok = ''
    embedding_dim = 100
    max_length = 250 # choose based on statistics, for example 150 to 200
    padding_type='post'
    trunc_type='post'

    # Tokenize the reviews using the loaded tokenizer
    sequences = tokenizer.texts_to_sequences([sentence])
    # Pad the sequences using the maximum length from training
    padded_sequences = pad_sequences(sequences, padding=padding_type, maxlen=max_length)

    # Predict the labels
    predicted_labels = model.predict(padded_sequences)
    
    # Convert the predicted labels to binary values (0 or 1)
    predicted_labels = np.round(predicted_labels).flatten()
    
    # Check if any element in the predicted labels is equal to 1
    if predicted_labels == 1:
        predicted_labels = "Positive"
    else:
        predicted_labels = "Negative"
    
    # print(predicted_labels)
    return render_template('activity6classification.html', review = sentence_output, predicted_labels=predicted_labels)
#--------------- Classification PART

@app.route('/About')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True)
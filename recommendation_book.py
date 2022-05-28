from flask import Flask, request, render_template
import os

# setting up template directory
TEMPLATE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')

app = Flask(__name__, template_folder=TEMPLATE_DIR, static_folder=TEMPLATE_DIR)
#app = Flask(__name__)

import pandas as pd
import numpy as np
from numpy import int64

import requests
import IPython.display as Disp

import sklearn
from sklearn.decomposition import TruncatedSVD





@app.route("/")
def hello():



    
    return TEMPLATE_DIR




books_df = pd.read_csv(r"C:\Users\admin\Desktop\recommendation_engine_project\templates\data_Set\books.csv")
ratings_df = pd.read_csv(r"C:\Users\admin\Desktop\recommendation_engine_project\templates\data_Set\ratings_algorithm.csv", encoding='UTF-8',  dtype={'user_id': int,'book_id':int, 'rating':int} )
books_df_2 = books_df[['original_title','authors']]
combined_books_df = pd.merge(ratings_df, books_df, on='book_id')
#creating pivot table
ct_df = combined_books_df.pivot_table(values='rating', index='user_id', columns='original_title', fill_value=0)
X = ct_df.values.T

#Creating SVD

SVD  = TruncatedSVD(n_components=20, random_state=17)
result_matrix = SVD.fit_transform(X)

#building correlation

corr_mat = np.corrcoef(result_matrix)
book_names = ct_df.columns
book_list = list(book_names)
isInitialized = True

print("done building recommendation engine")
print("ready for recommendation engine")



def getRecommendations(bookName):
    book_name_index = book_list.index(bookName)
    corr_book = corr_mat[book_name_index]
    recList = list(book_names[(corr_book<1.0) & (corr_book>0.8)])
    

    return books_df_2[books_df_2.original_title.isin(recList)]



@app.route("/rec",   methods=['GET', 'POST'])
def rec():
    query = '' 
    if(request.method == "POST"):
        print("inside post")
        print(str(request.form.get('query')))
        query = request.form.get('query')
        #print("the book name is " + query)
        recommendations = getRecommendations(query)
        
        #print(query)
        return render_template('rec.html', query=query, recommendations=recommendations.to_html())
    else:
        return render_template('rec.html', query="" ,recommendations="<<unknown>>")


    

if __name__ == "__main__":
    app.run(debug=True)


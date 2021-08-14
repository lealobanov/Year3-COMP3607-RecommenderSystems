import pandas as pd
import numpy as np
import math 
import sklearn

from operator import itemgetter

#/////////////ADDITIONAL DATA CLEANING AND MANIPULATION IN PREPARATION FOR CONTENT-BASED FILTERING///////////////////

def prepare_for_contentbased_filtering():
#Read in the cleaned dataset
    df = pd.read_csv("cleaned_dataset/cleanedOntario10.csv") 

    #Create integer user id and business id keys, which will be used to index the Numpy matrix
    df['user_id_key'] = pd.factorize(df.user_id)[0]
    df['business_id_key'] = pd.factorize(df.business_id)[0]

    df.drop_duplicates(subset=['business_id_key'], keep='first',inplace=True)

    #Create bag of words for each restaurant based on category (i.e. cuisine) and other qualitative features

    df["categories"].fillna("nan",inplace=True)

    df['bag_of_words'] = ''
    columns = df.columns
    for index, row in df.iterrows():
        words = ''
        #Parse categories bundle
        cats = row["categories"]
        if cats != 'nan':
            category_split = cats.split(',')
            for word in category_split:
                words += (word)
        else:
            #If nan found
            words = ""
        #Parse one-hot encoded features: from the intial dataset and also features that I created by combining relevant categories to resemble general cuisine groupings
        words += " "
        columns_toparse = ["RestaurantsTakeOut","NoiseLevel","GoodForKids","RestaurantsReservations","RestaurantsGoodForGroups","BusinessParking","RestaurantsPriceRange2","HasTV","Alcohol","BikeParking","RestaurantsDelivery","OutdoorSeating","WiFi","WheelchairAccessible","DriveThru","isUpscaleClassy","isRomanticIntimate","isCasual","isHipsterDiveyTrendy","gfDessert","gfLateNight","gfLunch","gfDinner","gfBreakfastBrunch","Ice Cream","Mexican Food","Asian Food","Chinese Food","Japanese Food","Italian Food","Cafe","Spanish Food","British Food","Latin American Food","Middle Eastern Food","Eastern European Food","Dessert Food","Bars and Entertainment"]
        for column in columns_toparse:
            val = row[column]
            if val == 1:
                words += column + " "
        df.loc[index, 'bag_of_words'] = words
    final = df.drop(df.columns.difference(['bag_of_words','business_id_key']), 1, inplace=False)

    #Save to CSV
    csv_name = "BOWOntario10.csv"
    final.to_csv(csv_name, index=False)

    return final

#/////////////////////////////////////////////////////////////////////////////////////////////////////////


#Find the 3 most highly rated restaurants by a particular user - these serve as the basis for content-based filtering
def find_highlyrated(df,user_id):
    by_userid = df.loc[df['user_id'] == user_id]
    by_userid = by_userid.sort_values(by=['review_stars'], ascending=False).head(4)
    business_ids = []
    for index, row in by_userid.iterrows():
        business = row["business_id"]
        business_ids.append(business)
    return business_ids[1:]

#Content-based filtering using keyword vectors generated by bag of words - however, a naive term frequency implementation is used here (not TF-IDF):
#https://towardsdatascience.com/yelp-restaurant-recommendation-system-capstone-project-264fe7a7dea1
#https://medium.com/@sumanadhikari/building-a-movie-recommendation-engine-using-scikit-learn-8dbb11c5aa4b

#TF-IDF vectorizer in sklearn documentation: https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html

from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import pairwise_distances

def content_based_filtering(df):
    #Initialize  CountVectorizer to generate a vector representation of restaurant features: this includes categories, attributes and other features I engineered myself to represent relvant grouping in the dataset

    #Naive term-frequency implementation
    #count = CountVectorizer()
    #count_matrix = count.fit_transform(df['bag_of_words'])

    #TF-IDF implementation
    tfidf_vectorizer=TfidfVectorizer(use_idf=True) 
    tfidf_matrix=tfidf_vectorizer.fit_transform(df['bag_of_words'])

    #Compute cosine similarities across all business ids
    cosine_sim = 1-pairwise_distances(tfidf_matrix, metric="cosine")
    return cosine_sim

#Retrive Yelp data set business id to dataframe index mapping, and vice versa
def get_title_from_index(index,df):
    return df[df["business_id_key"] == index]["business_id"].values[0]
def get_index_from_title(title,df):
    return df[df["business_id"] == title]["business_id_key"].values[0]

def content_based_byuser(main_df, user_id, similarities):

    #First generate highly rated ("favorite") restaurants by that user
    cb_input = find_highlyrated(main_df, user_id)
    restaurant_keys = []
    for restaurant in cb_input:
        indexval = get_index_from_title(restaurant,main_df)
        restaurant_keys.append(indexval)
    output = []
    for restaurant in restaurant_keys:
        similar_restaurants = list(enumerate(similarities[restaurant]))
        sorted_similar_restaurants = sorted(similar_restaurants,key=lambda x:x[1],reverse=True)[1:]
        i=0
        for element in sorted_similar_restaurants:
            busid = get_title_from_index(element[0],main_df)
            #Ensure only unique restaurants are added
            if busid not in output:
                output.append((busid, element[1]))
            i=i+1
            if i>=3:
                break
    #Sort the output before returning
    sorted_output = sorted(output,key=itemgetter(1),reverse=True)
    return output

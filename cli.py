import pandas as pd
import datetime

import collabfilter
import contentfilter

from operator import itemgetter


print("Initializing...")

#COLLAB-FILTERING SETUP

dfCollab = pd.read_csv("cleaned_dataset/collabRatingsKeyed.csv") 

#Read in the cleaned dataset
main = pd.read_csv("cleaned_dataset/cleanedOntario10.csv")
#Create integer user id and business id keys, which will be used to index the Numpy matrix
main['user_id_key'] = pd.factorize(main.user_id)[0]
main['business_id_key'] = pd.factorize(main.business_id)[0]


print("")
print("Constructing ratings matrix for collaborative filtering...")
#Construct ratings matrix - same for both memory based and model based approaches
ratings = collabfilter.construct_ratings_matrix(dfCollab)

print("")
print("Constructing prediction matrix for collaborative filtering...")
print("")
#Set up prediction matrix for collaborative filtering
collab_predictions = collabfilter.model_based_collab(ratings)

#CONTENT-BASED FILTERING SETUP

dfCB_bow = pd.read_csv("cleaned_dataset/BOWOntario10.csv") 

#Set up similarity matrix for content-based filtering
print("")
print("Constructing TF-IDF and corresponding similarity matrices for content-based filtering...")
cb_similarity = contentfilter.content_based_filtering(dfCB_bow)

print("")
print("Welcome to the Restaurant Recommender! Powered by the Yelp dataset.")

#CLI loop

def set_active_user():
    active_user = None
#Prompt to selet an active user
    print("")
    print("No active user currently selected.")
    print("")
    print("1 - From a preset list of sample users")
    print("2 - By providing a specific user id")
    print("")
    selection_method = input("Please indicate how you would like to define the active user (1/2): ")
    if selection_method == "1":
        print("")
        print("1 - jZrbuGRk92uWTu5kHbnDXQ")
        print("2 - ic-tyi1jElL_umxZVh8KNA")
        print("3 - q_HfkJHXgS2PmReenAOchQ")
        print("4 - _9WN_qmjbbRu6eFCMVXokw")
        print("5 - cESF_F2XYStfApl11wXVOA")
        print("6 - dG3E4PBsthCIqttNsj0edQ")
        print("7 - 26WgdHfEjWj4BrN-cUNhVw")
        print("8 - SZuxdORUn5-CRi2-9h-Idw")
        print("9 - SHcRMcO_dlmIsDATRqILQg")
        print("10 - AvsNwEUXnlns8TVjZkaMgQ")
        print("")
        dictusers = { "1":"jZrbuGRk92uWTu5kHbnDXQ", "2":"ic-tyi1jElL_umxZVh8KNA", "3":"q_HfkJHXgS2PmReenAOchQ", "4":"_9WN_qmjbbRu6eFCMVXokw", "5":"cESF_F2XYStfApl11wXVOA", "6": "dG3E4PBsthCIqttNsj0edQ", "7":"26WgdHfEjWj4BrN-cUNhVw", "8":"SZuxdORUn5-CRi2-9h-Idw", "9":"SHcRMcO_dlmIsDATRqILQg", "10":"AvsNwEUXnlns8TVjZkaMgQ"}
        while active_user == None:
            #Show a list of sample users 
            user = input("Please select a user from the list: ")
            if user in ["1","2","3","4","5","6","7","8","9","10"]:
                active_user = dictusers[user]
                return active_user

    else:
        #Prompt for user ID
        print("")
        while active_user == None:
            user = input("Please provide a user id: ")
            search = dfCollab[dfCollab['user_id'] == user]
            if len(search)>0:
                active_user = user
                return active_user

#Formatting recommendations for display on the screen 
def display_recommendations(restaurants, questionnaire_responses):
    print("")
    print("Your top 5 recommendations: ")
    #Iterate through each recommendation, showing relevant data AND explainability metrics
    i = 1
    for restaurant in restaurants:
        print ("")
        print("Recommendation #" +str(i))
        i += 1

        bus_id = restaurant[0]
        restaurant_name = main[main["business_id"] == bus_id]["business_name"].values[0]
        restaurant_address = main[main["business_id"] == bus_id]["address"].values[0]
        restaurant_city = main[main["business_id"] == bus_id]["city"].values[0]
        restaurant_state = main[main["business_id"] == bus_id]["state"].values[0]
        restaurant_postalcode = main[main["business_id"] == bus_id]["postal_code"].values[0]
        print("")
        print(str(restaurant_name))
        print(str(restaurant_address) + " " + str(restaurant_city) + " " + str(restaurant_state) + " " + str(restaurant_postalcode))
        print("")
        if restaurant[1] * 100 == restaurant[2]:
            filtering_type = "content based filtering"
            scoring_type = "Similarity score: "
        else:
            filtering_type = "collaborative filtering"
            scoring_type = "Predicted rating: "
        print("Recommendation generated by "+filtering_type+". "+scoring_type+str(restaurant[1])+" Overall score: "+str(restaurant[2]))

    #Construct explanation of questionnaire responses' impact on the recommendation rankings
    lines = []
    #Covid delivery
    if questionnaire_responses[0] == "y":
        lines.append("- You indicated a preference for restaurants with delivery, takeaway, or drive-thru services in light of COVID-19.")
    else:
        lines.append("- Prioritized restaurants with dine-in services. You did not indicate a preference for restaurants with delivery, takeaway, or drive-thru services in light of COVID-19.")

    #Cuisine
    cuisines = ""
    for cuisine in questionnaire_responses[1]:
        if cuisine == "1":
            #Mexican
            cuisines += "Mexican cuisine, "
        if cuisine == "2":
            #Italian
            cuisines += "Italian cuisine, "
        if cuisine == "3":
            #Asian (all)
            cuisines += "Asian cuisine, "
        if cuisine == "4":
            #Chinese
            cuisines += "Chinese cuisine, "
        if cuisine == "5":
            #Japanese
            cuisines += "Japanese cuisine, "
        if cuisine == "6":
            #Latin American
            cuisines += "Latin American cuisine, "
        if cuisine == "7":
            #Middle Eastern
            cuisines += "Middle Eastern cuisine, "
        if cuisine == "8": 
            #Desserts
            cuisines += "desserts, "
        if cuisine == "9":
            #Ice Cream
            cuisines += "ice cream, "
        if cuisine == "10":
            #Cafes
            cuisines += "cafes, "
        if cuisine == "11":
            #Bars+Entertainment
            cuisines += "bars and entertainment venues, "
    if cuisines != "":
        #Remove last comma
        cuisines = cuisines[:-2]
        lines.append("- Prioritized restaurants which align with your indicated preferences for " + cuisines + ".")

    #Occasion 
    occasion = questionnaire_responses[2]
    if occasion == "1":
        #Working - wifi
        lines.append("- You indicated a preference for a good working environment. Prioritized restaurants which have Wifi, quiet atmosphere, and are cafes.")
    if occasion == "2":
        #Casual meal
        lines.append("- You indicated a preference for a casual meal. Prioritized restaurants which have casual and trendy atmosphere, and are good for lunch.")
    if occasion == "3":
        #Date
        lines.append("- You indicated a preference for a date night. Prioritized restaurants which accept reservations, are quiet, good for dinner, and have a romantic, intimate, upscale, and classy atmosphere.")
    if occasion == "4":
        #Friends for drinks
        lines.append("- You indicated a preference for a night out with friends. Prioritized restaurants which are bars and entertainment venues with trendy/hipster atmosphere, are good for large groups and late night gatherings, and serve alcohol.")
    if occasion == "5":
        #Family
        lines.append("- You indicated a preference for a family outing. Prioritized restaurants which are good for kids.")


    #Parking
    if questionnaire_responses[3] == "y":
        lines.append("- Per your indication, prioritized restaurants with parking on-site.")

    #Time-sensitivity 
    tod = questionnaire_responses[4]
    if tod == "1":
        #Breakfast/brunch
        lines.append("- Per your preference for morning times, prioritized restaurants which are good for breakfast and brunch.")
    if tod == "2":
        #Lunch
        lines.append("- Per your preference for day times, prioritized restaurants which are good for lunch.")
    if tod == "3":
        #Dinner
        lines.append("- Per your preference for evening times, prioritized restaurants which are good for dinner.")
    if tod == "4":
        #Late night
        lines.append("- Per your preference for night times, prioritized restaurants which are good for late night outings.")

    #Further elaborate on explainability of predictions by considering questionnaire responses
    print("")
    print("Additional explainability of the recommendations, based on your indications in the questionnaire: " )
    print("")
    for line in lines:
        print(line)
    return

def knowledge_based(candidates,questionnaire):
    #Initialize empty scoring array
    qualitative_scores = []
    i = 0 
    while i < len(candidates):
        qualitative_scores.append([candidates[i][0], candidates[i][1], candidates[i][2], 0])
        i +=1 
    
    for entry in candidates:
        #Map to index in qualitative scores array 
        index = candidates.index(entry)
        bus_id = entry[0]

        #Covid delivery
        covid_delivery = questionnaire[0]
        #Check df if delivery or takeaway is true 
        doesTakeaway = main[main["business_id"] == bus_id]["RestaurantsTakeOut"].values[0]
        doesDelivery = main[main["business_id"] == bus_id]["RestaurantsDelivery"].values[0]
        doesDriveThru = main[main["business_id"] == bus_id]["DriveThru"].values[0]

        if doesTakeaway == 1 or doesDelivery == 1 or doesDriveThru == 1:
            if covid_delivery == "y":
                #Adjust the score
                qualitative_scores[index][3] += 1
        
        #Cuisines
        cuisines = questionnaire[1]

        for cuisine in cuisines:
            #Selecting 12 in the questionnaire indicated that the user finished picking cuisines they like early
            if cuisine != "12":
                if cuisines == "1":
                    #Mexican
                    isMexican = main[main["business_id"] == bus_id]["Mexican Food"].values[0]
                    if isMexican == 1:
                        qualitative_scores[index][3] += 1
                if cuisines == "2":
                    #Italian
                    isItalian = main[main["business_id"] == bus_id]["Italian Food"].values[0]
                    hasPizza = main[main["business_id"] == bus_id]["Pizza"].values[0]
                    if isItalian == 1 or hasPizza == 1:
                        qualitative_scores[index][3] += 1
                if cuisines == "3":
                    #Asian (all)
                    isAsian = main[main["business_id"] == bus_id]["Asian Food"].values[0]
                    if isAsian == 1:
                        qualitative_scores[index][3] += 1
                if cuisines == "4":
                    #Chinese
                    isChinese = main[main["business_id"] == bus_id]["Chinese Food"].values[0]
                    if isChinese == 1:
                        qualitative_scores[index][3] += 1
                if cuisines == "5":
                    #Japanese
                    isJapanese = main[main["business_id"] == bus_id]["Japanese Food"].values[0]
                    if isJapanese == 1:
                        qualitative_scores[index][3] += 1
                if cuisines == "6":
                    #Latin American
                    isLA = main[main["business_id"] == bus_id]["Latin American Food"].values[0]
                    if isLA == 1:
                        qualitative_scores[index][3] += 1
                if cuisines == "7":
                    #Middle Eastern
                    isME = main[main["business_id"] == bus_id]["Middle Eastern Food"].values[0]
                    if isME == 1:
                        qualitative_scores[index][3] += 1
                if cuisines == "8": 
                    #Desserts
                    isDessert = main[main["business_id"] == bus_id]["Dessert Food"].values[0]
                    gfDessert = main[main["business_id"] == bus_id]["gfDessert"].values[0]
                    if isDessert == 1 or gfDessert == 1:
                        qualitative_scores[index][3] += 1
                if cuisines == "9":
                    #Ice Cream
                    isIC = main[main["business_id"] == bus_id]["Ice Cream"].values[0]
                    if isIC == 1:
                        qualitative_scores[index][3] += 1
                if cuisines == "10":
                    #Cafes
                    isCafe = main[main["business_id"] == bus_id]["Cafe"].values[0]
                    if isCafe == 1:
                        qualitative_scores[index][3] += 1
                if cuisines == "11":
                    #Bars+Entertainment
                    isBE = main[main["business_id"] == bus_id]["Bars and Entertainment"].values[0]
                    if isBE == 1:
                        qualitative_scores[index][3] += 1

        #Occasion
        occasion = questionnaire[2]

        if occasion == "1":
            #Working - must have wifi
            hasWifi = main[main["business_id"] == bus_id]["WiFi"].values[0]
            if hasWifi == 1:
                qualitative_scores[index][3] += 1
            #Noise level quiet
            isQuiet = main[main["business_id"] == bus_id]["NoiseLevel"].values[0]
            if isQuiet == 1:
                qualitative_scores[index][3] += 1
            #Is a cafe
            isCafe = main[main["business_id"] == bus_id]["Cafe"].values[0]
            if isCafe == 1:
                qualitative_scores[index][3] += 1
                    
        if occasion == "2":
            #Casual
            isCasual = main[main["business_id"] == bus_id]["isCasual"].values[0]
            isHipsterDiveyTrendy = main[main["business_id"] == bus_id]["isHipsterDiveyTrendy"].values[0]
            gfL = main[main["business_id"] == bus_id]["gfLunch"].values[0]
            if isCasual == 1 or isHipsterDiveyTrendy == 1 or gfL == 1:
                qualitative_scores[index][3] += 1

        if occasion == "3":
            #Date night
            takesRes = main[main["business_id"] == bus_id]["RestaurantsReservations"].values[0]
            if takesRes == 1:
                qualitative_scores[index][3] += 1
            #Noise level quiet
            isQuiet = main[main["business_id"] == bus_id]["NoiseLevel"].values[0]
            if isQuiet == 1:
                qualitative_scores[index][3] += 1
            isUpscaleClassy = main[main["business_id"] == bus_id]["isUpscaleClassy"].values[0]
            isRomanticIntimate = main[main["business_id"] == bus_id]["isRomanticIntimate"].values[0]
            if isUpscaleClassy == 1 or isRomanticIntimate ==1 :
                qualitative_scores[index][3] += 1
            gfDinner = main[main["business_id"] == bus_id]["gfDinner"].values[0]
            if gfDinner == 1:
                qualitative_scores[index][3] += 1

        if occasion == "4":
            #Friends for drinks
            gfLateNight = main[main["business_id"] == bus_id]["gfLateNight"].values[0]
            if gfLateNight == 1:
                qualitative_scores[index][3] += 1
            gfGroups = main[main["business_id"] == bus_id]["RestaurantsGoodForGroups"].values[0]
            if gfGroups == 1:
                qualitative_scores[index][3] += 1
            servesAlcohol = main[main["business_id"] == bus_id]["Alcohol"].values[0]
            if servesAlcohol == 1:
                qualitative_scores[index][3] += 1
            isBE = main[main["business_id"] == bus_id]["Bars and Entertainment"].values[0]
            if isBE == 1:
                qualitative_scores[index][3] += 1
            isHipsterDiveyTrendy = main[main["business_id"] == bus_id]["isHipsterDiveyTrendy"].values[0]
            if isHipsterDiveyTrendy == 1:
                qualitative_scores[index][3] += 1

        if occasion == "5":
            #Family outing
            gfKids = main[main["business_id"] == bus_id]["GoodForKids"].values[0]
            if gfKids == 1:
                qualitative_scores[index][3] += 1
    
        #Parking
        parking = questionnaire[3]
        #Check df if hasParking is true
        hasParking = main[main["business_id"] == bus_id]["BusinessParking"].values[0]
        if hasParking == 1:
            if parking == "y":
                #Adjust the score
                qualitative_scores[index][3] += 1
    
        #Time_sensitvity 
        tod = questionnaire[4]
        if tod == "1":
            #Breakfast and brunch
            gfBB = main[main["business_id"] == bus_id]["gfBreakfastBrunch"].values[0]
            if gfBB == 1:
                qualitative_scores[index][3] += 1
        elif tod == "2":
            #Lunch
            gfL = main[main["business_id"] == bus_id]["gfLunch"].values[0]
            if gfL == 1:
                qualitative_scores[index][3] += 1
        elif tod == "3":
            #Dinner
            gfD = main[main["business_id"] == bus_id]["gfDinner"].values[0]
            if gfD == 1:
                qualitative_scores[index][3] += 1
        else:
            #TOD == 4, late night
            gfLN = main[main["business_id"] == bus_id]["gfLateNight"].values[0]
            if gfLN == 1:
                qualitative_scores[index][3] += 1

    #Sort qualitative scores in descending order
    sorted_scores = sorted(qualitative_scores,key=itemgetter(3),reverse=True)
    #Return the top 5 scoring restaurants
    output = sorted_scores[:5]
    return output

def start_recommendation(active_user):
    print("")
    print("The current active user is " + active_user)
    print("")
    print("To get started, please answer a few questions so we can better tailor our recommendations to your current needs.")  
    print("")
    print("The current metropolitan area is set to Toronto, Ontario.") 
    #Questionnaire
    print("")
    #Covid - check if user prefers delivery or eat-in service
    delivery_method = None
    while delivery_method == None:
        delivery = input("Given COVID-19 we understand you may not prefer dine-in services. Would you prefer home delivery or takeaway? (y/n) ")
        if delivery == "y":
            delivery_method = 'y'
        elif delivery == 'n':
            delivery_method = 'n'

    #Cuisine preferences
    cuisines = []
    print("")
    print("Do you enjoy any of the following cuisines? You may select up to 3.") 
    print("")
    print("1 - Mexican Food")
    print("2 - Italian Food")
    print("3 - Asian Food (all)")
    print("4 - Asian Food - Chinese")
    print("5 - Asian Food - Japanese")
    print("6 - Latin American Food")
    print("7 - Middle Eastern Food")
    print("8 - Desserts")
    print("9 - Ice Cream")
    print("10 - Cafes")
    print("11 - Bars and Entertainment")
    print("12 - Nothing else interests me")

    while len(cuisines) < 3:
        print("")
        cuisine = input("Select a cuisine from the list: ")
        if cuisine in ["1","2","3","4","5","6","7","8","9","10","11","12"]:
            if cuisine not in cuisines:
                if cuisine == "12":
                    #User has no preference/doesn't like any of the cuisines in the list
                    break
                else:
                    cuisines.append(cuisine)

    #Occasion           
    print("")
    print("What's the occasion?")
    print("")
    print("1 - Catching up on work. I need wifi!")
    print("2 - Just a casual meal")
    print("3 - Date night")
    print("4 - Meeting friends for drinks")
    print("5 - Family outing")
    print("")

    occasion = None
    while occasion == None:
        select_occasion = input("Select an occasion from the list: ")
        if select_occasion in ["1","2","3","4","5"]:
            occasion = select_occasion

    #Parking
    print("")
    parking = None
    while parking == None:
        need_parking = input("Do you need parking at the venue? (y/n) ")
        if need_parking == "y":
            parking = need_parking
        elif need_parking == "n":
            parking = need_parking

    #Time-sensitivity
    time_now = datetime.datetime.now().time()
    time_split = str(time_now).split(":")
    hour = int(time_split[0])
    timeofday = ""
    if hour < 12 and hour > 3:
        #Breakfast
        timeofday = "1"
        tod = "breakfast and brunch"
    elif hour >=12 and hour <= 16:
        #Lunch
        timeofday = "2"
        tod = "lunch"
    elif hour >= 15 and hour < 21:
        #Dinner
        timeofday = "3"
        tod = "dinner"
    else:
        #Late night
        timeofday = "4"
        tod = "late night venues"

    print("")
    print("It's currently approximately " + str(time_split[0])+ ":00. The time of day suggests it is time for " + tod +".")
    print("")
    agree = input("Do you agree with this selection? If not, we can recommend restaurants for a different time of day. (y/n) ")
    if agree == 'y':
        time_sensitivity = timeofday
    else:
        print("")
        print("1 - Breakfast/brunch")
        print("2 - Lunch")
        print("3 - Dinner")
        print("4 - Late night")
        print("")
        time_sensitivity = None
        while time_sensitivity == None:
            time_picked = input("Select a desired dining time: ")
            if time_picked in ["1","2","3","4"]:
                time_sensitivity = time_picked
    print("")
    print("Thank you for your responses! Generating recommendations...")
    print("")

    #Perform collaborative-based filtering on the user id

    #Convert user id to user key 
    user_key = collabfilter.get_index_from_userid(active_user, dfCollab)
    collab_business_predictions = collabfilter.model_based_prediction(user_key, collab_predictions)

    #Convert business predictions to business ids, with prediction scores - returns 9 candidates
    collab_busids=[]
    for index in collab_business_predictions:
        busid = contentfilter.get_title_from_index(index[0],dfCollab)
        collab_busids.append((busid,index[1]))

    #Perform content-based filtering (item-based) on the user id  - returns 9 candidates
    contentbased_business_predictions = contentfilter.content_based_byuser(main, active_user, cb_similarity)

    #Mixed hybrid system
    #Take a union of the candidates, sorted by rank score

    #Establish a ranking system for recommendations - need to standardize a metric to reflect similarity score (content-based) and prediction score (collaborative)
    #Retain the original scoring metric for explainability purposes

    #Score out of ~100 
    #Collaborative filtering prediction score *20 (previously in range 0 to 5)
    collab_standardized_scores = []
    for entry in collab_busids:
        collab_standardized_scores.append((entry[0], entry[1], entry[1]*20))
    
    #Content based similarity score * 100 (previously in range 0 to 1)
    cb_standardized_scores = []
    for entry in contentbased_business_predictions:
        cb_standardized_scores.append((entry[0], entry[1], entry[1]*100))

    duplicate_business_ids = []
    #Remove any non-unique business IDs (retain the higher score)
    for entry in collab_standardized_scores:
        for entry2 in cb_standardized_scores:
            if entry[0] == entry2[0]:
                #Ensure the duplicate is not added twice
                if (entry, entry2) not in duplicate_business_ids:
                    duplicate_business_ids.append((entry, entry2))
    
    #Retain the higher score of duplicate businesses 
    if len(duplicate_business_ids)>0:
        for pair in duplicate_business_ids:
            score1 = pair[0][2]
            score2 = pair[1][2]
            if score1 > score2:
                #Collab score is higher - remove from content-based
                cb_standardized_scores = cb_standardized_scores.remove(pair[1])
            else:
                #Content-based score is higher - remove from collab
                collab_standardized_scores = collab_standardized_scores.remove(pair[0])
        
    #Take union of both lists and sort
    combined_scores = collab_standardized_scores + cb_standardized_scores
    combined_scores = sorted(combined_scores,key=itemgetter(2),reverse=True)
    top_combined = combined_scores[:12]
    
    #Further filter the results using questionnaire responses: knowledge-based approach - passing on the top 12
    #Coallate the user's questionnaire responses
    questionnaire = [delivery_method,cuisines,occasion,parking,time_sensitivity]
    final_output = knowledge_based(top_combined, questionnaire)

    return final_output, questionnaire

active_user = None
first = True
while True:
    if first:
        first = False
    else:
        #Option to quit the recommender after at least one recommendation is made
        print("")
        continue_program = input("Would you like to receive another recommendation? (y/n) ")
        print("")
        if continue_program == "n":
            break

    if active_user == None:
        active_user = set_active_user()
        out, questionnaire = start_recommendation(active_user)
        display_recommendations(out, questionnaire)
    else:
        print("The current active user is " + active_user + "." )
        print("")
        change_active = input("Would you like to change the active user? (y/n) ")
        if change_active == "y":
            active_user = set_active_user()
            out, questionnaire = start_recommendation(active_user)
            display_recommendations(out, questionnaire)
        else:
            out, questionnaire = start_recommendation(active_user)
            display_recommendations(out, questionnaire)




from keras.models import load_model
from numpy import array as np
import numpy as nump
from time import sleep

#TODO: check input
#TODO: celery to pause ml moves (for experience)

def predictModel(categories):

    #using the trained model to predict

    #categories=["Graphics cards", "Monitors", "Processors", "Mouse", "Keyboards"]

    num_cat=[]
    prediction=[]

    #TODO: change this to search for the tags/categories? (Data which user clicks on)
    for cat in categories:
        if cat=="Graphics cards":
            num_cat.append("1")
        if cat=="Monitors":
            num_cat.append("2")
        if cat=="Processors":
            num_cat.append("3")
        if cat=="Mouse":
            num_cat.append("4")
        if cat=="Keyboards":
            num_cat.append("5")
        else:
            num_cat.append("6")

    while len(num_cat)<18:
        num_cat.append("0")
    
    num_cat_numpy=[]
    num_cat_numpy.append(num_cat)
    num_cat_numpy=np(num_cat_numpy)

    num_cat_numpy = nump.reshape(num_cat_numpy, (num_cat_numpy.shape[0], num_cat_numpy.shape[1]))

    print("0: ", num_cat_numpy.shape[0], "1: ", num_cat_numpy.shape[1])

    prediction=model.predict(num_cat_numpy).argmax()
    print("Prediction number: ", prediction)
    
    predict=NumberToPredCat(prediction)
    #predict=InvertPred(cat)

    return InvertPred(predict)

def NumberToPredCat(num):
        if num==1:
            return "Graphics cards"
        elif num==2:
            return "Monitors"
        elif num==3:
             return "Processors"
        elif num==4:
            return "Mouse"
        elif num==5:
            return "Keyboards"
        else:
            return "Shannon"
        
def InvertPred(cat):
    if cat=="Graphics cards":
        return "Graphics cards"
    elif cat=="Monitors":
        return "Monitors"
    elif cat=="Processors":
        return "Processors"
    elif cat=="Mouse":
        return "Mouse"
    elif cat=="Keyboards":
        return "Keyboards"

def getPrediction(categories):
    return predictModel(categories)

model = load_model('model.h5')

categories =[]

#TODO: use this as manual for now, maybe change to an input, but update as clicked tag in the future
user_category = "Processors"
#user_category = input("Your component ")

categories.append(user_category)

print ("Your category: ", "".join(str(c) for c in categories))

predicted_cat=getPrediction(categories)

#TODO: use celery
sleep(0.5)

print ("Prediction: ", predicted_cat)


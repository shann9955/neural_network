from keras.models import load_model
from numpy import array as np

#using the trained model to predict
model = load_model('model.h5')

categories=["Graphics cards", "Monitors", "Processors", "Mouse", "Keyboards"]

num_cat=[]
num_cat_numpy=[]

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

while len(num_cat)<18:
    num_cat.append("0")

num_cat_numpy.append(num_cat)
num_cat_numpy=np(num_cat_numpy)

prediction=model.predict(num_cat_numpy).argmax()
print(prediction)

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
    
def InvertPred(cat):
    if cat=="Graphics cards":
        return "GC"
    elif cat=="Monitors":
        return "M"
    elif cat=="Processors":
        return "P"
    elif cat=="Mouse":
        return "M"
    elif cat=="Keyboards":
        return "K"
      
move=NumberToPredCat(prediction)
move=InvertPred(cat)
print(move)

#TODO: check input
#TODO: celery to pause ml moves (for experience)



import tkinter as tk
from sklearn.ensemble import RandomForestRegressor
import pickle
import numpy as np

#load X_train, X_test, y_train, y_test from train_test_data pickle
pfile = open('train_test_data', 'rb')
X_train = pickle.load(pfile)
X_test = pickle.load(pfile)
y_train = pickle.load(pfile)
y_test = pickle.load(pfile)

regressor = RandomForestRegressor()
regressor.fit(X_train, y_train.ravel())
y_pred = regressor.predict(X_test)
  
# tkinter GUI
root= tk.Tk()

canvas1 = tk.Canvas(root, width = 500, height = 350)
canvas1.pack()

# user input - views
label1 = tk.Label(root, text='                   Views ::')
canvas1.create_window(100, 100, window=label1)

entry1 = tk.Entry(root)
canvas1.create_window(270, 100, window=entry1)

# user input - Dislikes
label2 = tk.Label(root, text='         Dislikes :: ')
canvas1.create_window(120, 120, window=label2)

entry2 = tk.Entry(root)
canvas1.create_window(270, 120, window=entry2)

# user input - Comments
label3 = tk.Label(root, text='Comments ::')
canvas1.create_window(140, 140, window=label3)

entry3 = tk.Entry(root)
canvas1.create_window(270, 140, window=entry3)

def values():
    global viewsCount
    viewsCount = float(entry1.get())
 
    global dislikes   
    dislikes =  float(entry2.get())
  
    global comments  
    comments =  float(entry3.get())

    X_input=np.array([[dislikes, comments,viewsCount]])
    X_input.reshape(1,-1)
    pred = int(regressor.predict(X_input))
   

    label_Prediction = tk.Label(root, text= pred, bg='sky blue')
    canvas1.create_window(270, 280, window=label_Prediction)


#Creating one button which will trigger prediction logic.
button1 = tk.Button (root, text='      Predict      ',command=values, bg='green', fg='white', font=11)
canvas1.create_window(270, 220, window=button1)
 
root.mainloop()
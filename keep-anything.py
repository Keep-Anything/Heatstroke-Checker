from tkinter import *
import customtkinter
from PIL import ImageTk, Image
from typing import *
import numpy as np
import pandas as pd
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt

# reading external csv file
# missing values are automatically read as nan
testDF = pd.read_csv(r'data\\heat2.csv', names=['temp', 'na_lvl', 'pulse', 'bp', 'resp_rate', 'spo2_per', 'perdict_h'])
#print(testDF)
# loading training data set which has been cleaned now
mydata = loadtxt(r'data\\heat2.csv', delimiter=",")
#print(mydata)

x = mydata[:, 0:6]
y = mydata[:, 6]
# selecting type of model
model = Sequential()
model.add(Dense(256, input_dim=6, activation="tanh"))
model.add(Dense(128, activation="tanh"))
model.add(Dense(64, activation="tanh"))
model.add(Dense(32, activation="relu"))
model.add(Dense(16, activation="relu"))
model.add(Dense(1, activation="sigmoid"))
# compiling model with appropriate parameters
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
# fitting the model with 100 epochs and batch size 40
model.fit(x, y, epochs=500, batch_size=48)
# getting the exact accuracy in number
_, accuracy = model.evaluate(x, y)
print("Test Set Accuracy: ", accuracy)

window = customtkinter.CTk()
window.geometry("390x700")
window.title("Heatstroke Checker")

#background color
window.bg = customtkinter.CTkLabel(window, fg_color= "#E2C3AB", text = '')
window.bg.pack(fill = 'both', expand = 'yes')

#Title
window.title = customtkinter.CTkLabel(window, text = "CURRENT HEAT STRESS", font = ("Franklin Gothic Medium", 30), fg_color="#E2C3AB", text_color='black')
window.title.place(x=195, y=20, anchor = CENTER)

#side icon
window.side_image = Image.open("images\\icon.png")
window.side_image = window.side_image.resize((200, 200))
icon1 = ImageTk.PhotoImage(window.side_image)
window.side_panel = customtkinter.CTkLabel(window, image = icon1, fg_color='#E2C3AB', text = '')
window.side_panel.image = icon1
window.side_panel.place(x=215, y=108)

#input title
window.input_title = customtkinter.CTkLabel(window, text = "Vital Information", font=("Franklin Gothic Medium", 25), text_color='black',fg_color='#E2C3AB')
window.input_title.place(x=12, y=300)

#frame below
window.text_frame = customtkinter.CTkFrame(window, fg_color='#ECDD7B',border_color='#E2C3AB', bg_color='#E2C3AB', border_width=2, width=248, height=298)
window.text_frame.place(x=10, y = 335, anchor = NW)
window.text_frame.grid_propagate(False)
window.text_frame.pack_propagate(False)

#text inside frame
window.label1 = customtkinter.CTkLabel(window.text_frame, text = "Core Temperature",font = ("Franklin Gothic Medium", 20), text_color='black')
window.label2 = customtkinter.CTkLabel(window.text_frame, text = "Sodium Level",font = ("Franklin Gothic Medium", 20), text_color='black')
window.label3 = customtkinter.CTkLabel(window.text_frame, text = "Pulse",font = ("Franklin Gothic Medium", 20), text_color='black')
window.label4 = customtkinter.CTkLabel(window.text_frame, text = "Blood Pressure",font = ("Franklin Gothic Medium", 20), text_color='black')
window.label5 = customtkinter.CTkLabel(window.text_frame, text = "Respiratory Rate",font = ("Franklin Gothic Medium", 20), text_color='black')
window.label6 = customtkinter.CTkLabel(window.text_frame, text = "Oxygen Level",font = ("Franklin Gothic Medium", 20), text_color='black')

window.label1.place(x=20, y = 30, anchor = W)
window.label2.place(x=20, y = 77, anchor = W)
window.label3.place(x=20, y = 124, anchor = W)
window.label4.place(x=20, y = 171, anchor = W)
window.label5.place(x=20, y = 218, anchor = W)
window.label6.place(x=20, y = 265, anchor = W)

#input frame
window.input_frame = customtkinter.CTkFrame(window, fg_color='#ECDD7B',border_color='#E2C3AB', bg_color='#E2C3AB', border_width=2, width=111, height=298)
window.input_frame.place(x=268, y = 335, anchor = NW)
window.temp_input = customtkinter.CTkEntry(window.input_frame, width = 70, height = 20, fg_color='white', text_color='black')
window.sodium_input = customtkinter.CTkEntry(window.input_frame, width = 70, height = 20, fg_color='white', text_color='black')
window.pulse_input = customtkinter.CTkEntry(window.input_frame, width = 70, height = 20, fg_color='white', text_color='black')
window.bp_input = customtkinter.CTkEntry(window.input_frame, width = 70, height = 20, fg_color='white', text_color='black')
window.resp_input = customtkinter.CTkEntry(window.input_frame, width = 70, height = 20, fg_color='white', text_color='black')
window.o2_input = customtkinter.CTkEntry(window.input_frame, width = 70, height = 20, fg_color='white', text_color='black')

window.temp_input.place(x=18, y = 30, anchor = W)
window.sodium_input.place(x=18, y = 77, anchor = W)
window.pulse_input.place(x=18, y = 124, anchor = W)
window.bp_input.place(x=18, y = 171, anchor = W)
window.resp_input.place(x=18, y = 218, anchor = W)
window.o2_input.place(x=18, y = 265, anchor = W)

#Function when the predict button is pressed
def assign_to_dict():

    list1 = [float(window.temp_input.get()), float(window.sodium_input.get()), float(window.pulse_input.get()), float(window.bp_input.get()), float(window.resp_input.get()),float(window.o2_input.get())]
    df = pd.DataFrame(list1).transpose()
    prediction(window, df)

#Function when the stress level is pressed
def new():

    days=['Sunday','Monday','Tuesday','Wed','Thur','Fri','Sat']

    numbers=[0.8,0.5,0.3,0.4,0.6,0.1,0.2]

    plt.plot(days,numbers,color='red',linewidth=2,marker='o',markerfacecolor='grey',markersize=7)
    plt.xlabel('Days')
    plt.ylabel('Heat-stress level')
    plt.title('My stress levels')

    plt.show()


#predict button
window.enter_data = customtkinter.CTkButton(window, text = "Predict", width=170, height=30, border_color='#E2C3AB', bg_color='#E2C3AB', command=assign_to_dict)
window.enter_data.place(x=208, y=646, anchor=NW)

#stress button
window.stress_button = customtkinter.CTkButton(window, text = "Stress Level", width=170, height=30, border_color='#E2C3AB', bg_color='#E2C3AB', command=new)
window.stress_button.place(x=12, y=646, anchor=NW)

#prediction label
window.prediction_title = customtkinter.CTkLabel(window, text = "Prediction", font=("Franklin Gothic Medium", 22), text_color='black',fg_color='#E2C3AB')
window.prediction_title.place(x=14, y=114)

window.prediction_box = customtkinter.CTkLabel(window, text = 'Enter Data', font = ("Franklin Gothic Medium", 40), text_color='black', fg_color='#E2C3AB', bg_color='#E2C3AB', width=207, height=149)        
window.prediction_box.place(x=12, y=144)

window.prediction_bool = customtkinter.CTkLabel(window, text = '', font = ("Franklin Gothic Medium", 20),text_color='black', fg_color='#E2C3AB', bg_color='black')
window.prediction_bool.place(x=115, y=260, anchor=CENTER)

#finding the prediction based on the input through the model
def prediction(window, testdata):
    predictions = model.predict(testdata)
    rounded = np.round(predictions, 2)
    print(testdata, "predicts", rounded[0])

    value = rounded.item() * 100
    value = round(value, 2)
    text_result = f"{value}%"
    print(text_result)

    #write the prediction to the GUI
    window.prediction_box = customtkinter.CTkLabel(window, text = text_result, font = ("Franklin Gothic Medium", 40), text_color='black', fg_color='#E2C3AB', bg_color='#E2C3AB', width=207, height=149)
    window.prediction_box.place(x=12, y=144)
    #write whether the user has heatstroke or not
    if rounded.item() > 0.5:
        window.prediction_bool = customtkinter.CTkLabel(window, text = 'Heatstroke', font = ("Franklin Gothic Medium", 20),text_color='black', fg_color='#E2C3AB', bg_color='black')
        window.prediction_bool.place(x=115, y=260, anchor=CENTER)
    else:
        window.prediction_bool = customtkinter.CTkLabel(window, text = 'No Heatstroke', font = ("Franklin Gothic Medium", 20),text_color='black', fg_color='#E2C3AB', bg_color='black')
        window.prediction_bool.place(x=115, y=260, anchor=CENTER)

window.mainloop()

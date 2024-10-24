from flask import Flask, render_template, jsonify, request,current_app
import socket
import pickle
import ast
import csv
import io
import joblib 
import pandas as pd
import threading
import serial
import time
import pygame
from timeit import default_timer as timer
from sklearn.preprocessing import LabelEncoder
import logging
from collections import Counter
from threading import Thread
from flask import Flask, render_template, url_for, redirect
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin, login_user, LoginManager, login_required, logout_user, current_user
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField, SelectField
from wtforms.validators import InputRequired, Length, ValidationError
from flask_bcrypt import Bcrypt
from flask import Flask, request, redirect, url_for
from facestrain import train_classifier
from faces import perform_facial_recognition
import os
import cv2
from apscheduler.schedulers.background import BackgroundScheduler
from threading import Thread
import time
from flask_login import LoginManager
from flask import session
from collections import Counter
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.schedulers.base import STATE_STOPPED
import random
import smtplib
from threading import Lock

otp_lock = Lock()
face_lock = Lock()
client_socket=None
server_socket=None
serverRun=False
stopServerFlag = threading.Event()
stopServerFlagLock = threading.Lock()
breakTime = None
user_id=0
facial_recognition_thread_flag=False
otp_thread_flag = False
usernameGlob = None
scheduler = BackgroundScheduler()
scheduler2 = BackgroundScheduler()
scheduler_ = BackgroundScheduler()
current_otp=0
most_common = ""
userName = ""
systemType = "A"
facial_recognition_thread = None
otp_thread = None

current_directory = os.getcwd()
TEMP_IMAGE_FOLDER = os.path.join(current_directory, 'images')
IMAGE_FOLDER = os.path.join(current_directory, 'images')
MODEL_PATH = os.path.join(current_directory, 'recognizers', 'face-trainner.yml')

app = Flask(__name__,template_folder='Templates')
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
app.debug=True
logging.basicConfig(level=logging.DEBUG)
current_directory = os.getcwd()
dbpath = os.path.join(current_directory, "database2.db")
# app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///D:/TCS_YF/Research/OpenCV-Python-Series-master/OpenCV-Python-Series-master/src/database2.db'
app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{dbpath}'
app.config['SECRET_KEY'] = 'randomKeyForNow'
app.config['SERVER_NAME'] = 'localhost:5000'
db = SQLAlchemy(app)
bcrypt = Bcrypt(app)



class User(db.Model,UserMixin):
    id=db.Column(db.Integer,primary_key=True)
    username=db.Column(db.String(20),nullable=False,unique=True)
    password=db.Column(db.String(80),nullable=False)

class UserDetails(db.Model):
    id=db.Column(db.Integer,primary_key=True)
    user_id = db.Column(db.Integer,db.ForeignKey('user.id'),nullable=False)
    full_name = db.Column(db.String(50), nullable=False)
    occupation = db.Column(db.String(50), nullable=True)
    birthday = db.Column(db.Integer, nullable=True)
    gender = db.Column(db.String(10), nullable=True)
    user = db.relationship('User',backref='details',uselist=False)

class registerForm(FlaskForm):
    username=StringField(validators=[InputRequired(),Length(
        min=4,max=20)],render_kw={"placeholder":"Username"})
    
    password=StringField(validators=[InputRequired(),Length(
        min=4,max=20)],render_kw={"placeholder":"Password"})
    
    submit = SubmitField("Register")

    def validate_username(self,username):
        existing_username = User.query.filter_by(username=username.data).first()
        if existing_username:
            raise ValidationError("This username is already taken. Please try another one.")

class loginForm(FlaskForm):
    username=StringField(validators=[InputRequired(),Length(
        min=4,max=20)],render_kw={"placeholder":"Username"})
    
    password=StringField(validators=[InputRequired(),Length(
        min=4,max=20)],render_kw={"placeholder":"Password"})
    
    submit = SubmitField("Login")

class personalDetails(FlaskForm):
    first_name = StringField(validators=[InputRequired(),Length(
        min=4,max=20)],render_kw={"placeholder":"First Name"})
    last_name = StringField(validators=[InputRequired(),Length(
        min=4,max=20)],render_kw={"placeholder":"Last Name"})
    occupation = StringField(validators=[InputRequired(),Length(
        min=4,max=20)],render_kw={"placeholder":"Job Role"})
    day_of_birth = SelectField('Day', choices=[(str(i), str(i)) for i in range(1, 32)])
    month_of_birth = SelectField('Month', choices=[(str(i), str(i)) for i in range(1, 13)])
    year_of_birth = SelectField('Year', choices=[(str(i), str(i)) for i in range(1900, 2023)])
    gender = SelectField('Gender', choices=[('male', 'Male'), ('female', 'Female'), ('other', 'Other')])
    # Add other fields as needed
    submit = SubmitField('Submit')

@app.route('/')
def home():
    return render_template('startPage.html')

@app.route('/personalDetails/<int:user_id>',methods=['GET','POST'])
def showPersonalDetails(user_id):
    user = User.query.get(user_id)
    form = personalDetails(user_id=user_id)
    if form.validate_on_submit():
        new_details = UserDetails(user_id=user.id,
                                  full_name=form.first_name.data + " " + form.last_name.data,
                                  birthday=form.day_of_birth.data + "/" + form.month_of_birth.data + "/" + form.year_of_birth.data,
                                  occupation=form.occupation.data,
                                  gender=form.gender.data)
        db.session.add(new_details)
        db.session.commit()

        return redirect(url_for('login'))
    
    print('PERSONAL')
    print(form.validate())
    print(form.errors)
    return render_template('personalDetails.html',form=form)


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

@app.route('/SystemC',methods=['GET','POST'])
def systemC():
    global systemType
    systemType="C"
    systemCFace()
    shutDownScheduler2()
    return render_template('SystemC.html')

def shutDownScheduler2():
    global otp_thread_flag
    global systemType
    global otp_thread
    global facial_recognition_thread_flag
    if scheduler2.state != STATE_STOPPED:
        scheduler2.shutdown()
        otp_thread.join()
        facial_recognition_thread_flag = False
        facial_recognition_thread
        print("Scheduler2 has been shut down")

@app.route('/FinishPage',methods=['GET','POST'])
def finishPage():
    return render_template('FinishPage.html')

@app.route('/SystemB',methods=['GET','POST'])
def systemB():
    global otp_thread_flag
    global systemType
    global otp_thread
    global facial_recognition_thread_flag
    if scheduler.state != STATE_STOPPED:
        scheduler.shutdown()
        facial_recognition_thread_flag = False
        facial_recognition_thread
        print("Scheduler has been shut down")
    if not otp_thread_flag:
        print("Inside SystemA route")
        otp_thread_flag = True
        otp_thread = Thread(target=run_otp_thread,args=[session.get('username', None)])
        print("Thread for OTP will be started")
        otp_thread.start()
    
    systemType = "B"
    return render_template('SystemB.html')

def run_otp_thread(username):
    global otp_thread_flag
    global scheduler2
    # scheduler = BackgroundScheduler()
    scheduler2.add_job(OTPGenerate, 'interval', seconds=45)  # Adjust the interval as needed
    scheduler2.start()
    print("OTP Thread started, scheduler started")

#solely sending otp
@app.route('/sendOTP', methods=['GET','POST'])
def sendOTP():
    OTPGenerate()
    return jsonify({'State': "Successful"})


def OTPGenerate():
    global current_otp
    otp_lock.acquire()
    try:
        otp=''.join([str(random.randint(0,9)) for i in range(4)])
        server=smtplib.SMTP('smtp.gmail.com',587)
        server.starttls()
        password='jjgd ochd wrti lmks'
        server.login('usertestingauthentication@gmail.com',password)
        msg='Hello, Your OTP for User Testing for System' + systemType + ' is '+str(otp)
        #session['otp'] = otp
        current_otp = otp
        print(f"OTP generated will be sent as {otp}")
        server.sendmail('usertestingauthentication@gmail.com','usertestingauthentication@gmail.com',msg)
        print("Email sent")
        server.quit()
    finally:
        otp_lock.release()

@app.route('/checkOTP', methods=['GET'])
def check_for_prompt():
    global current_otp
    show_prompt = current_otp
    print(f"Prompt request received, otp is currently {show_prompt}")
    return jsonify({'OTP': show_prompt})

@app.route('/checkFace', methods=['GET'])
def check_for_face():
    print("In check face check_for_face rn")
    if userName == None or userName != most_common:
        print("No face")
        shutdown_scheduler()
        with app.app_context():
            return jsonify({'Face': None})
    elif userName != most_common:
        print("Does not match")
        with app.app_context():
            return jsonify({'Face':"NoMatch"})
            return redirect(url_for('logout'))
    return jsonify({'Face': userName})

@app.route('/loginPage',methods=['GET','POST'])
def login():
    #global usernameGlob
    form = loginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        #user = User.query.filter_by(username=form.username.data).first()
        print("Retrieved user:", user)

        #usernameGlob = form.username.data
        if user:
            if bcrypt.check_password_hash(user.password,form.password.data):
                login_user(user)
                session['username'] = form.username.data
                print("Login successful")
                return redirect(url_for('systemA'))
    print("User not found")
    return render_template('loginPage.html',form=form)

@app.route('/SystemCFace', methods=['GET','POST'])
def systemCFace():
    global facial_recognition_thread_flag
    print("IN SYSTEM C FACIAL RECOGNITION")
    print(f"State of facial recognition thread flag {facial_recognition_thread_flag}")
    if not facial_recognition_thread_flag:
        print("Inside C route")
        facial_recognition_thread_flag = True
        facial_recognition_thread = Thread(target=run_facial_recognition_thread,args=[session.get('username', None)])
        print("Thread will be started")
        facial_recognition_thread.start()


@app.route('/SystemA', methods=['GET','POST'])
def systemA():
    global facial_recognition_thread_flag
    global systemA 
    global facial_recognition_thread

    #Start the facial recognition thread only if it's not already running
    if not facial_recognition_thread_flag:
        print("Inside SystemA route")
        facial_recognition_thread_flag = True
        facial_recognition_thread = Thread(target=run_facial_recognition_thread,args=[session.get('username', None)])
        print("Thread will be started")
        facial_recognition_thread.start()

    systemA = "A"
    return render_template('SystemA.html')

def run_facial_recognition_thread(username):
    global facial_recognition_thread_flag
    global scheduler
    #username = session.get('username',None)
    print(f"Username got from session is: {username}")
    # scheduler = BackgroundScheduler()
    print(f"Scheduler.running {scheduler.running}")
    print(f"Scheduler {scheduler}")
    if not scheduler.running:
        scheduler = BackgroundScheduler()
    scheduler.add_job(perform_facial_recognition2, 'interval', seconds=40,args=[username])  # Adjust the interval as needed
    scheduler.start()
    print("Thread started, scheduler started")

def perform_facial_recognition2(username):
    global most_common
    global userName
    userName = username
    nameList = []
    print("Inside perform_facial_recognition")
    face_lock.acquire()
    try: 
        start = timer()
        x=perform_facial_recognition()
        nameList.append(x)
        x=perform_facial_recognition()
        nameList.append(x)
        x=perform_facial_recognition()
        nameList.append(x)
        # name = perform_facial_recognition()
        for i in nameList:
            print("elements of nameList are:")
            print(i)

        counter = Counter(nameList)
        most_common,count = counter.most_common(1)[0]
        print("Face detected by algorithm MOST COMMONLY is:")
        print(most_common)
        print(f"Username is: {username}")
    finally:
        face_lock.release()

@app.route('/schedulerShutdown', methods=['[POST]'])
def shutdown_scheduler():
    global scheduler
    if scheduler and scheduler.state != STATE_STOPPED:
        scheduler.shutdown()
        facial_recognition_thread.join()
        
@app.route('/logout', methods=['GET', 'POST'])
@login_required
def logout():
    print("User has been logged out")
    logout_user()
    print("Redirecting to login")
    return redirect(url_for('login'))

@app.route('/OTP', methods=['GET', 'POST'])
def OTPHandler():
    global otp_thread_flag
    global current_otp
    data=request.json
    print(f"Received OTP data: {data}")
    state = data['state']
    OTP = data['input']
    print(f"state is: {state} and otp is {OTP}")
    if state == "Cancelled":
        print("IS CANCELLED")
        if scheduler2.state != STATE_STOPPED:
            scheduler2.shutdown()
        otp_thread_flag = False
        otp_thread.join()
        print("OTP WAS NONE SO SHUT IT DOWN")
        print("Scheduler has been shut down")
        logout()
        return redirect(url_for('logout'))
    if OTP is None:
        if scheduler2.state != STATE_STOPPED:
            scheduler2.shutdown()
            print("Scheduler has been shut down")
        print("OTP WAS NONE SO SHUT IT DOWN")
        otp_thread_flag = False
        otp_thread.join()
        logout()
        return redirect(url_for('logout'))
    else:
        if OTP!=current_otp:
            return redirect(url_for('logout'))
        
@app.route('/schedulerShutdown_', methods=['[POST]'])
def shutdown_scheduler_():
    global scheduler_
    # if scheduler_ and scheduler_.state != STATE_STOPPED:
    #     scheduler_.shutdown()
    #     facial_recognition_thread.join()



@app.route('/registerPage',methods=['GET','POST'])
def register():
    form = registerForm()
    if form.validate_on_submit():
        hashed_password = bcrypt.generate_password_hash(form.password.data)
        new_user = User(username=form.username.data,password=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        print(f"New user ID: {new_user.id}")
        print(f"Username : {form.username.data}")
        #return redirect(url_for('camera_page', user_id=new_user.id))
        return redirect(url_for('camera_page', form=form, username=form.username.data))

    print("Form validation failed.")
    print(form.validate())
    print(form.errors)
    return render_template('registerPage.html',form=form)

@app.route('/camera/<username>', methods=['GET', 'POST'])
def camera_page(username):
    print("Usnermae in camera_page username ",username)
    # Pass the username to the HTML template
    return render_template('camera.html', username=username)


def capture_pictures(Username, photo_count):

    print(f"Capturing pictures for {Username}, Photo Count: {photo_count}")

    # Try to capture a frame
    webcam = cv2.VideoCapture(0)

    key = cv2.waitKey(1)
    while True:
        try:
            check, frame = webcam.read()
            if not check:
                print("Failed to capture frame.")
                break

            # print(check)  # prints true as long as the webcam is running
            # print(frame)  # prints matrix values of each frame
            cv2.imshow("Capturing", frame)

            key = cv2.waitKey(1)
            if key == ord('s'):
                # If the frame is captured successfully, save it
                user_image_dir = os.path.join(IMAGE_FOLDER, Username.lower())
                os.makedirs(user_image_dir, exist_ok=True)
                image_path = os.path.join(user_image_dir, f"{photo_count}.png")
                cv2.imwrite(image_path, frame)
                print(f"Frame captured successfully. Saved at: {image_path}")
                break


                temp_image_dir = os.path.join(TEMP_IMAGE_FOLDER, username)
                os.makedirs(temp_image_dir, exist_ok=True)
                image_path = os.path.join(temp_image_dir, f"{photo_count}.png")
                cv2.imwrite(image_path, frame)
                print(f"Frame captured successfully. Saved at: {image_path}")
                break

        except KeyboardInterrupt:
            print("Turning off camera.")
            break

    # Release the webcam
    webcam.release()
    cv2.destroyAllWindows()

    if frame is not None:
        print(f"Frame captured successfully. Saved at: {image_path}")
    else:
        print("Failed to capture frame.")



def create_user_folder(username):
    # Create a subfolder for the user within the 'images' directory
    user_image_dir = os.path.join(IMAGE_FOLDER, username.lower())
    os.makedirs(user_image_dir, exist_ok=True)
    return user_image_dir

# @app.route('/camera', methods=['POST'])
# def capture_picture():
#     username = request.args.get('username')
#     photo_count = request.args.get('photoCount')
#     print(f"Received POST request for {username}, Photo Count: {photo_count}")

#     # Capture and save the image
#     capture_pictures(username, photo_count)

#     if photo_count < 3:  # Adjust the condition based on your requirements
#         return jsonify({"status": "success", "message": "Picture captured successfully!"})
#     else:
#         user_image_dir = os.path.join(IMAGE_FOLDER, username)
#         for i in range(4):
#             temp_image_path = os.path.join(TEMP_IMAGE_FOLDER, username, f"{username}_{i}.png")
#             user_image_path = os.path.join(user_image_dir, f"{username}_{i}.png")
#             os.rename(temp_image_path, user_image_path)

#         return jsonify({"status": "success", "message": "All pictures captured successfully!"})



@app.route('/cam', methods=['POST'])
def capture_picture():
    data=request.json
    print(f"Received POST request data: {data}")

    username2=data['Username']
    photo_count=data['photoCount']
    print(f"Received POST request for {username2}, Photo Count: {photo_count}")
    #photo_count = request.args.get('photo_count', type=int, default=0)

    if photo_count <= 5:
        capture_pictures(username2, photo_count)
        return jsonify({"status": "success", "message": "Picture captured successfully!"})

    elif photo_count ==6:

        # Perform model retraining and move images to user folder
        train_classifier()
        return jsonify({"status": "success", "message": "Classifier done"})
    # else:
    #     return jsonify({"status": "success", "message": "Redirect to SystemA"})
    return redirect(url_for('login'))  # Redirect to the login page or another page


if __name__ == '__main__':
    print("CURRENT DIRECTORY IS:", os.getcwd())
    current_directory = os.getcwd()
    TEMP_IMAGE_FOLDER = os.path.join(current_directory, 'images')
    IMAGE_FOLDER = os.path.join(current_directory, 'images')
    MODEL_PATH = os.path.join(current_directory, 'recognizers', 'face-trainner.yml')

# Print the paths to verify
    print(f"TEMP_IMAGE_FOLDER: {TEMP_IMAGE_FOLDER}")
    print(f"IMAGE_FOLDER: {IMAGE_FOLDER}")
    print(f"MODEL_PATH: {MODEL_PATH}")
    with app.app_context():
        db.create_all()
    #app.run(debug=True)

    # thread = Thread(target = threaded_function, args = (10, ))
    # thread.start()

    app.run()
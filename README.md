# Facial Recognition vs TOTP authentication vs Hybrid System



## About
This system aids the research Optimal Passwordless Continuous Authentication Measures for Remote Employees during usability testing. The research aims to compare the trade-offs between intrusive
and non-intrusive authentication measures whilst considering security as well as user-convenience. It aims to do so by developing a web-app system that implements these authentication measures
for testing through usability testing and performance metrics. The system further homogenises various factors (such as programming environment, usability testing sample size, etc.) that aid in analysing a more unbiased result. 

This system is the web-app that has been developed. It mimics login and register processes as well as a productive environment by means of various tasks. Furthermore, it consists of 3 systems: System A (facial recognition), System B (TOTP based authentication) and System C (Hybrid Authentication). On each page after login, one authentication measure will be deployed in the background while the user "works". There are no right or wrong answers to the tasks, it simply helps users get a feel of what each authentication measure in a productive environment is like.


## Prerequisites
For this system to run successfully, you will need to install Python and an IDE to run the same in. To install Python successfully, this tutorial can be followed:
```
https://phoenixnap.com/kb/how-to-install-python-3-windows
```
You will need to install OpenCV, and Flask which can be followed here:
```
https://www.geeksforgeeks.org/how-to-install-opencv-for-python-in-windows/

https://flask.palletsprojects.com/en/3.0.x/installation/
```
For this system you will also need the facial recognition library. The same can be found here:

```
https://github.com/ageitgey/face_recognition
```

In order for the HTML and CSS files to load appropriately, you will need to download the Bogart fonts online, however, these can easily be replaced with whichever fonts you desire.

## Running
To interact with the web application, you will need to first need to unzip the `Remote-Employees-and-Continuous-Authentication-Web-Application` folder. Then, you must access the `OpenCV-Python-Series-master` folder, following which the `src` folder and run the `webApp.py` file within the same. This can be found at:
```
OpenCV-Python-Series-master\src\webApp.py
```

Following this, you will be required to open a browser of your choice and run localhost on it, as the instructions in your terminal will display. 



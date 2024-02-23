import cv2
import dlib
import math
import RPi.GPIO as GPIO
from time import sleep
import time
from gpiozero import Buzzer
from picamera.array import PiRGBArray
from picamera import PiCamera

GPIO.setmode(GPIO.BCM)

#on met le seuil qui dicte si es yeux sont ouvert ou non
# valeur trouver avec des tests
SEUIL_DE_FERMETURE = 0.25

#les pin utilisés pour la partie physique du projet
bz = Buzzer(17)
BUTTON_PIN = 27
GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down = GPIO.PUD_UP)
#fonction pour calculer la distance entre deux points
def distance(point1 ,point2):
    return math.sqrt((point1[0]-point2[0])**2+(point1[1]-point2[1])**2)

#fonction pour calculer la fermeture des yeux
def ratio_fermeture_yeux(eye_points, facial_landmarks):
# Les coordonnés en (x,y) des yeux
    P0 = (facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y)
    P1 = (facial_landmarks.part(eye_points[1]).x, facial_landmarks.part(eye_points[1]).y)
    P2 = (facial_landmarks.part(eye_points[2]).x, facial_landmarks.part(eye_points[2]).y)
    P3 = (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y)
    P4 = (facial_landmarks.part(eye_points[4]).x, facial_landmarks.part(eye_points[4]).y)
    P5 = (facial_landmarks.part(eye_points[5]).x, facial_landmarks.part(eye_points[5]).y)
    
    A = distance(P1,P5)
    B = distance(P2,P4)
    C = distance(P0,P3)
    
    ratio = (A+B)/(2*C)
    return ratio

#utilisation de l'algorithmes haarcascade
profile_cascade = cv2.CascadeClassifier('haarcascade_profileface.xml')
visage_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#les performances de la caméras qui sont adapté pour un Pi caméra version 2
camera = PiCamera()
camera.resolution = (1280,720)
camera.framerate = 90
raw_capture = PiRGBArray(camera, size = (1280,720))
time.sleep(0.1)

#algorithmes de détection du visage selon Dlib
detecteur = dlib.get_frontal_face_detector()

#cartographie du visage selon les 68 points
predicteur = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

#assignation des points pour chaque oeil selon les 68 points du face landmarks
oeil_gauche_landmarks = [36, 37, 38, 39, 40, 41]
oeil_droit_landmarks = [42, 43, 44, 45, 46, 47]

#acces à la caméra grâce
for frame in camera.capture_continuous(raw_capture, format="bgr", use_video_port=True):
# arret de secours grace au boutton
    if GPIO.input(BUTTON_PIN) == GPIO.LOW:
        print("arret du systeme")
        bz.off()
        sleep(5)
    frame = frame.array
    
    #variabble pour manipuler les detection et ensuite pouvoir leur assigner des etats
    gauche = False
    droite = False
    devant = False
    #detection du profile et du visage
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    profile = profile_cascade.detectMultiScale(gray, 1.3, 5)
    visage = visage_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in profile:
        if w > 0:
            gauche = True
        #rectangle autour du visage lorsque detecter
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
    for(x,y,w,h) in visage:
        if w > 0:
            devant = True
            
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        
#pour pouvoir detecter l'autre cote du visage
flipped = cv2.flip(gray, 1)
profile = profile_cascade.detectMultiScale(flipped, 1.3, 5)
for (x,y,w,h) in profile:
    if w>0:
        droite = True
    cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
#condition pour que ce soit une situation dangereuse
if ((gauche == True) or (droite == True)) and (devant == False):
    b=0
    print("attente profile du visage")
    #temps d'attente pour ensuite verifier la deuxieme condition
    t_end = time.time()+2
    while time.time()<t_end:
        b=b+1
        
#deuxieme verification de la condition apres que le temps soit passe
if (devant == False):
    print("ALARME")
    bz.on()
    sleep(0.5)
    cv2.putText(frame,"ATTENTION",(10,50), cv2.FONT_HERSHEY_SIMPLEX,2,(255,255,255),2,cv2.LINE_AA)
#condition pour situation de perte de conscience au volant
if (gauche == False) and (droite == False) and (devant == False):
    a = 0
    print("attente visage")
    t_end = time.time()+1
    while time.time()<t_end:
        if (devant == False):
        print("ALARME")
        bz.on()
    cv2.putText(frame,"ATTENTION",(10,50), cv2.FONT_HERSHEY_SIMPLEX,2,(255,255,255),2,cv2.LINE_AA)
else:
    bz.off()
    
#detection du visage grace a dlib
faces,_,_ = detecteur.run(image = frame, upsample_num_times = 0, adjust_threshold = 0.0)
for face in faces:
    #isolation des yeux
    landmarks = predicteur(frame, face)
    #calcul de l'ouverture des yeux
    ratio_oeil_gauche = ratio_fermeture_yeux(oeil_gauche_landmarks, landmarks)
    ratio_oeil_droit = ratio_fermeture_yeux(oeil_droit_landmarks, landmarks)
    #moyenne des deux yeux
    ratio_fermeture = (ratio_oeil_gauche+ratio_oeil_droit)/2
#Situation de danger par rapport a la fermeture des yeux
    if (ratio_fermeture < SEUIL_DE_FERMETURE):
    d = 0
    print("attente ferme yeux")
    while (d<100):
    d=d+1
    if(ratio_fermeture < 0.25):
    print("ALARME")
    bz.on()
    sleep(0.3)
    cv2.putText(frame,"ATTENTION",(10,50), cv2.FONT_HERSHEY_SIMPLEX,2,(255,255,255),2,cv2.LINE_AA)
else:
    bz.off()
    
#affichage de la camera
cv2.imshow('detection', frame)
key = cv2.waitKey(1)
raw_capture.truncate(0)
if key == 27:
    break

GPIO.cleanup()
cv2.destroyAllWindows()
bz.off()
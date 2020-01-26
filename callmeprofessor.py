#http://www.lfd.uci.edu/~gohlke/pythonlibs/#numpy
#pip install pypiwin32
#pip install numpy-1.11.3+mkl-cp27-cp27m-win32.whl
import speech_recognition
import pyttsx
from pprint import pprint
import cv2
import cv2
import time
import os
import sys
import csv
import imutils
import numpy as np
import datetime
from google import search
#response = urllib.urlopen ('https://en.wikipedia.org/wiki/Google').read()
#json = m_json.loads(response)
#results = json [ 'responseData' ] [ 'results' ]
import sys
import wikipedia
from multiprocessing.pool import ThreadPool
pool = ThreadPool(processes=4)

def diffImg(t0, t1, t2):
  d1 = cv2.absdiff(t2, t1)
  d2 = cv2.absdiff(t1, t0)
  return cv2.bitwise_and(d1, d2)

cam = cv2.VideoCapture(0)
time.sleep(2)

winName = "Movement Indicator"
cv2.namedWindow(winName, cv2.WINDOW_NORMAL)
currentworkingdirectory = str(os.getcwd()).replace("\\","\\")

# Read three images first:

frame = cam.read()[1]
gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
gray = cv2.GaussianBlur(gray, (21, 21), 0)
firstFrame = gray

#speech_engine = pyttsx.init('sapi5')
#speech_engine.setProperty('rate', 150)

speech_engine = pyttsx.init()
speech_engine.setProperty('rate', 160)

def speak(text):
    speech_engine.say(text)
    speech_engine.runAndWait()


recognizer = speech_recognition.Recognizer()

def listen():
    zresult = ""
    with speech_recognition.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

	try:
            #zresult = recognizer.recognize_sphinx(audio)
            #zresult = recognizer.recognize_google(audio, show_all=True)
            recognizer.operation_timeout = 5
            #zresult = recognizer.recognize_google(audio)
            zresult = recognizer.recognize_google(audio)
            #print("Sphinx thinks you said " + zresult)
            print("Google Speech Recognition results:")
            zresult = zresult.replace("nice to meet you what's your name","").strip()
            pprint(zresult) 
            speak("you speak "+zresult)
            #print("Microsoft Bing Voice Recognition thinks you said " + recognizer.recognize_bing(audio, key=BING_KEY, show_all=True))

            speak_list = zresult.split()
            if any("what" in s for s in speak_list):
                speak("let me search for you")
                speak( wikipedia.summary(zresult.replace("what is ","").replace("what are ","").replace("what was ","").replace("what were ",""), sentences=2).encode(sys.stdout.encoding, errors='replace'))
                #for url in search("wiki " + zresult.replace("what is ","").replace("what are ","").replace("what was ","").replace("what were ",""), tld='co.uk', lang='uk', stop=5):
                    #print(url)
            if any("which" in s for s in speak_list):
                speak("is there any choice for me to choose?")
            if any("where" in s for s in speak_list):
                speak("i am searching this place")
            if any("do you" in s for s in speak_list):
                speak("i do not answer true or false question")
            if any("is there" in s for s in speak_list):
                speak("i do not know the environment")

            #return recognizer.recognize_google(audio)
            #return recognizer.recognize_sphinx(audio)
            # or: return recognizer.recognize_google(audio)
	except speech_recognition.UnknownValueError:
            print("Exception : Could not understand audio")
            zresult = "Could not understand audio"
	except speech_recognition.RequestError as e:
	    print("Recog Error; {0}".format(e))
            zresult = "Could not understand audio"
    return zresult


start = datetime.datetime.now()
done = datetime.datetime.now()

def greetingfunc(myname):
    # initialize the first frame in the video stream
    firstFrame = None
    framecount = 0
    while framecount < 30:
        frame = cam.read()[1] 	
        #frame = imutils.resize(frame, width=900)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (201, 201), 0)
	if firstFrame is None:
		firstFrame = gray
		continue
	frameDelta = cv2.absdiff(firstFrame, gray)
        thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
	thresh = cv2.dilate(thresh, None, iterations=2)
        (cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
	# loop over the contours
	for c in cnts:
            if cv2.contourArea(c) > 3:
                (x, y, w, h) = cv2.boundingRect(c)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
	                 
        if frame is not None:
            if myname is not None:
                cv2.imwrite(myname + ".png", frame)
            else:
                cv2.imwrite("guest.png", frame)
            #cv2.imwrite("debug1.png", frame)
            key = cv2.waitKey(1) & 0xFF
            #cv2.imshow("Thresh", thresh)
            #cv2.imshow("Frame Delta", frameDelta)
            framecount = framecount + 1


def getpathofface(nameparam):
    currentworkingdirectory = str(os.getcwd()).replace("\\","\\")
    iserror = False
    headervalues = []
    with open(str(currentworkingdirectory)+"\\faceconfig.csv", 'rb') as csvfile:
        reader = csv.reader(csvfile)
        headersource = reader.next()

    header = []
    for headerm in headersource:
        header.append(headerm)
        #header.append("[" + headerm + "]")

    linenumber = 0
    yourname = ""
    #print(str(currentworkingdirectory)+"\\faceconfig.csv")
    #print(headersource)
    with open(str(currentworkingdirectory)+"\\faceconfig.csv", 'rb') as csvfile:
        reader = csv.DictReader(csvfile)
        #print(header)
        for row in reader:
            notvalidfield = False
            linenumber = linenumber + 1
            for hh in headersource:
                if len(row[hh].strip()) <= 0:
                    notvalidfield = True
            if notvalidfield == True:
                print("line number " + str(linenumber) + " one of field is empty")
            if iserror == False and notvalidfield == False:
                try:
                    if row["name"].strip() == nameparam:
                        #print("find your name")
                        yourname = row["path"].strip()
                        #print("path : " + row["path"].strip())
                except:
                    print "Unexpected error:", sys.exc_info()[0]
                    iserror = True
    return yourname

alreadygreeting = False
while True:
  if alreadygreeting == False:
      speak("Nice to meet you, what's your name?")
      myname = ""
      try:
          #async_result = pool.apply_async(listen)
          #myname = async_result.get(timeout=10)
          thread = Thread(target=listen)
          thread.start()
          thread.join()
      except:
          print "Unexpected error:", sys.exc_info()[0]
          print "Unexpected error:", sys.exc_info()[1]
          alreadygreeting = False
          issucceed = False
      if (myname is None) or "Could not understand audio" in str(myname).strip():
          issucceed = False
          alreadygreeting = False
      else:
          issucceed = True
          alreadygreeting = True
          alreadygreeting = True
      numberoftrial = 0
      while issucceed == False:
          try:
              print("could you speak again")
              speak("could you speak again")
              issucceed = False
              myname = ""
              #async_result = pool.apply_async(listen)
              print("run more time")
              #myname = async_result.get(timeout=10)
              thread = Thread(target=listen)
              thread.start()
              thread.join()
          except:
              print "Unexpected error:", sys.exc_info()[0]
              print "Unexpected error:", sys.exc_info()[1]
              alreadygreeting = False
              issucceed = False
          if (myname is None) or "Could not understand audio" in str(myname).strip():
              issucceed = False
              alreadygreeting = False
          else:
              issucceed = True
              alreadygreeting = True
          numberoftrial = numberoftrial + 1
      greetingfunc(myname)
      

  frame = cam.read()[1]
  if firstFrame is None:
      firstFrame = gray
      continue

  facepath = getpathofface(myname)
  
  if os.path.isfile(facepath):
      img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
      template = cv2.imread(facepath,0)
      w, h = template.shape[::-1]

      res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)
      threshold = 0.64
    
      loc = np.where( res >= threshold)
      pt = [(0,0)]
    
      while not zip(*loc[::-1]):
          threshold = threshold - 0.02
          loc = np.where( res >= threshold)

      counter = 1
      #print("threshold="+str(threshold))   
      for pt2 in zip(*loc[::-1]):
          elapsed = done - start
          if threshold > 0.3:
              #print("reach")
              #cv2.rectangle(frame, pt2, (pt2[0] + w, pt2[1] + h), (0,0,255), 2)
              if counter == 1 and elapsed.total_seconds() == 0:
                  speak("Hello, Professor Lee Ho Yeung")
                  speak("Anything i can help")
                  async_result = pool.apply_async(listen) 
                  start = datetime.datetime.now()
                  print("first time run")
              if elapsed.total_seconds() > 10:
                  speak("Hello, Professor Lee Ho Yeung")
                  speak("Anything i can help")
                  async_result = pool.apply_async(listen) 
                  start = datetime.datetime.now()
                  print("after first time run")
          if counter > 200:
              counter = 2
              #print(str(elapsed.total_seconds()))
              done = datetime.datetime.now()
          counter = counter + 1

      if frame is not None:
          cv2.imshow( winName, frame)
          #key = cv2.waitKey(1) & 0xFF

  else:
      alreadygreeting = False
      
  key = cv2.waitKey(10)
  if key == 27:
    cv2.destroyWindow(winName)
    break

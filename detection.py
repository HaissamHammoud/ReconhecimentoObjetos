import cv2 as cv
import glob
#############################################################################
#este algoritmo busca as imagens jpg contidas na pasta e mostra os rostos---#
#contidos em cada imagem na pasta imagens-----------------------------------#
#############################################################################
def detectAndDisplay(frame):  #define uma função
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)  #transforma o frame em cinza
    frame_gray = cv.equalizeHist(frame_gray)   # equaliza histogram da imagem
    #Detecta faces
    
    faces = face_cascade.detectMultiScale(frame_gray,1.058,15)  #encontra a face das pessoas
    for (x,y,w,h) in faces:
        center = (x + w//2, y + h//2)
        frame = cv.ellipse(frame, center, (w//2, h//2), 0, 0, 360, (255, 0, 255), 4) # desenha o rosto na imagem
        faceROI = frame_gray[y:y+h,x:x+w]

    faces = catface_cascade.detectMultiScale(frame_gray,1.05,13)  #encontra a face das pessoas
    for (x,y,w,h) in faces:
        center = (x + w//2, y + h//2)
        frame = cv.ellipse(frame, center, (w//2, h//2), 0, 0, 360, (0, 0, 255), 4) # desenha o rosto na imagem
        faceROI = frame_gray[y:y+h,x:x+w]
    frame = cv.resize(frame,(900,700))    
    cv.imshow('Capture - Face detection', frame) #mostra a imagem para o usuario
    cv.waitKey(0)
face_cascade = cv.CascadeClassifier("haarcascade_frontalface_default.xml")
catface_cascade = cv.CascadeClassifier("catface.xml")

#img = cv.imread("imgfacedec\imagens\faces1.png")
#detectAndDisplay(img)
images=glob.glob("imagens\*")

for i in images:
    img = cv.imread(i)
    
    detectAndDisplay(img)
    
cv.waitKey(0)
cv.destroyAllWindows()

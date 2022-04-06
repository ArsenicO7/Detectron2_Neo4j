from Detector import *

detector = Detector(model_type="OD")

detector.onImage("/home/arsenic/Documents/complete_detectron2/images/3.jpg")

#test comment
#for x in range(1 , 5):
    #detector.onImage("home/arsenic/Documents/complete_detectron2/images/{}.jpg".format(x))
    #print (detector.list[dict])
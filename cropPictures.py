import cv2
import glob
import numpy as np

h = 170 #138
w = 106
y = 83
x = [44, 146, 342, 446]

char_count = [0] * 36
char_path = lambda x: '/home/fizzer/enph353_cnn_lab/data/' + x + '.png'

for plate_path in glob.glob('/home/fizzer/enph353_cnn_lab/pictures/*.png'):
    plate_val = plate_path[-8:-4]
    plate_img = cv2.cvtColor(cv2.imread(plate_path), cv2.COLOR_BGR2GRAY)

    for i in range(4):
        char = cv2.resize(plate_img[y:y+h, x[i]:x[i]+w], None, fx=0.5, fy=0.5) #cropping and scaling it down
        ind = ord(plate_val[i]) # ascii value of the letter or number
        ind = ind - 48 if ind < 58 else ind - 55 # less than 58 its a number
        char_count[ind] = char_count[ind] + 1
        cv2.imwrite(char_path(plate_val[i] + '%03d' % char_count[ind]), char) # write char w its value 

print(char_count)
print(np.sum(char_count))

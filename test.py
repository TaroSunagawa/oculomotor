import cv2
import numpy as np
import time
import brica
import tensorflow as tf 

last = cv2.imread('./log/sequence/0.jpg')
now = cv2.imread('./log/sequence/1.jpg')

last_hist = cv2.calcHist(last,  [0], None, [256], [0, 256])
now_hist = cv2.calcHist(now,  [0], None, [256], [0, 256])
result = cv2.compareHist(last_hist, now_hist, 0)
print(str(result))

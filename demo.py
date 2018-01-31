
from ctpnport import *
from crnnport import *
import matplotlib.pyplot as plt
import sys

#ctpn
text_detector = ctpnSource()
#crnn
model,converter = crnnSource()

timer = Timer()
print('input exit break')
while 1 :
    if sys.version_info >= (3, 0):
        im_name = input('\nplease input file name: ')
    else:
        im_name = raw_input("\nplease input file name: ")

    if im_name == "exit":
       break
    im_path = "./img/" + im_name
    im = cv2.imread(im_path)
    if im is None:
        continue

    img, text_recs = getCharBlock(text_detector, im)
    timer.tic()
    crnnRec(model, converter, img, text_recs)
    print("Time: %f" % timer.toc())

    cv2.waitKey(0)
    cv2.destroyAllWindows()


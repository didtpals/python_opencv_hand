import cv2 as cv
import numpy as np

# hand_img 변수에 imread 함수를 사용하여 이미지를 넣어줌
hand_img = cv.imread("hand.jpg") 

# 위 코드에서 가져온 이미지는 RGB(BGR)인데 cvtColor 함수를 사용해서 YCrCB로 만들어줌
# 그 이유는 이미지에서 손에 해당하는 피부색 부분만을 추출하기 위함임

# RGB(BGR) 이란 색을 표현하는 방식 ( 각 색의 값은 0 ~ 255 사이의 값으로 표시하는데 값이 높아질수록 색의 빛이 밝아지는 원리)
# YCbCr 도 색을 표현하는 방식 색차(색깔의 차이 정도) 정보를 확인하여 Cb, Cr로 분리하여 표현 || YCbCr도 각각(0 ~ 255 사이의 값을 가짐)
# Y = 휘도(밝기) 색차 성분(Cb, Cr)
ycrcb_hand_img = cv.cvtColor(hand_img, cv.COLOR_BGR2YCrCb)

# RGB 에서 ycrcb 로 바뀐 이미지의 피부색 부분만을 추출해야하는데 Cb와 Cr을 입력해야함
# Cb와 Cr은 각각의 색차를 입력해야하는데 보통 피부색의(살색에 해당하는 색) 값은 Cb: 77 ~ 127, Cr: 133 ~ 173 임
# 이미지에서 특정 부분이 Cb와 Cr 사이의 색 값에 해당하면 흰색으로 만들고 아니면 모두 검은색으로 만들어줌

# inRange 함수를 사용해 min, max 값을 입력해 색을 추출
check_color_hand_img = cv.inRange(ycrcb_hand_img, np.array([0, 133, 77]), np.array([255, 173, 127]))

# 원래 이미지에 외곽선을 그려주는 코드 
#  https://deep-learning-study.tistory.com/231

# 외곽선 검출을 해주는 findContours 함수 
# 첫 번쨰 인자 = 이미지, 두 번째 인자 = 외곽선 검출 모드, 세 번째 = 외곽선을 단순화 하는 방법(모드)
contours, hierarchy = cv.findContours(check_color_hand_img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

# 외곽선을 그려주는 drawContours 함수 
# 기존 이미지에 검출한 외곽선을 그려야함
# 첫 번째 인자 = 기존에 호출한 RGB 이미지, 두 번쨰 인자 = 위에서 외곽선을 검출해준 값을 가지고있는 변수 
# # 세 번째 인자 = 특정 숫자를 지정하면 외곽선을 그림 (-1은 모든 외곽선을 그림)
# # 네 번째 인자 = 어느 색으로 그릴지(RGB) 
# # 다섯 번째 인자 = 선의 두께
cv.drawContours(hand_img, contours, -1, (255,255,0), 1) 

cv.imshow("hand", hand_img)

cv.waitKeyEx(0)
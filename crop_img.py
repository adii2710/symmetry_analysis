# from rembg import remove

# class RemImgBackground:
#     def __init__(self):
#         self.img=None
#         self.out_img=None
    
#     def remBg(self, img):
#         self.img=img
#         out=remove(self.img)
#         # out.save('images/out.png')
#         return out

import cv2

class RemImgBackground:
    def __init__(self):
        self.img=None
        self.out_img=None
    
    def remBg(self, img):
        self.img=img
        # Create a BackgroundSubtractor
        bg_subtractor = cv2.createBackgroundSubtractorMOG2()

        # Apply the BackgroundSubtractor
        fg_mask = bg_subtractor.apply(self.img)

        # Create a mask by thresholding the foreground mask
        _, mask = cv2.threshold(fg_mask, 127, 255, cv2.THRESH_BINARY)

        # Apply the mask to the image
        result = cv2.bitwise_and(self.img, self.img, mask=mask)

        return result
from rembg import remove

class RemImgBackground:
    def __init__(self):
        self.img=None
        self.out_img=None
    
    def remBg(self, img):
        self.img=img
        out=remove(self.img)
        # out.save('images/out.png')
        return out
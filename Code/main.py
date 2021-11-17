#Import
from PIL import Image

#Constants
Af_tr_images = 510
As_tr_images = 510

newSize = 50

def cropImages():
    print("Cropping images ...")
    for number in range(1, Af_tr_images + 1):
        print("Opening Af tr image: " + str(number))
        im = Image.open("dataset/train/African/af_tr" + str(number) + ".jpg")

        resized_image = im.resize((newSize, newSize))
        resized_image.convert('RGB').save("dataset/train/Resized_Images/African/African_" + str(number) + ".jpg")

    for number in range(1, As_tr_images + 1):
        print("Opening As tr image: " + str(number))
        im = Image.open("dataset/train/Asian/as_tr" + str(number) + ".jpg")

        resized_image = im.resize((newSize, newSize))
        resized_image.convert('RGB').save("dataset/test/Resized_Images/Asian/Asian_" + str(number) + ".jpg")

    print("Images cropped!");

def loadImages():
    print("Loading images ...")
    train_set_x = []
    train_set_y = []

    #Load african images
    for number in range(1, ):
        print("test")

    print("Images loaded!")

if __name__ == '__main__':
    cropImages();
    #loadImages();



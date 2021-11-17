# Import
from PIL import Image


# Constants
newSize = 50


def cropimages():
    print("Cropping images ...")
    for number in range(1, 421):
        print("Opening Af tr image: " + str(number))
        im = Image.open("Dataset/Train/African/af_tr" + str(number) + ".jpg")

        resized_image = im.resize((newSize, newSize))
        resized_image.convert('RGB').save("Dataset/Train/Resized_Images/African/African_" + str(number) + ".jpg")

    for number in range(1, 91):
        print("Opening Af te image: " + str(number))
        im = Image.open("Dataset/Train/African/Af_te" + str(number) + ".jpg")

        resized_image = im.resize((newSize, newSize))
        resized_image.convert('RGB').save("Dataset/Train/Resized_Images/African/African_" + str(number + 420) + ".jpg")

    for number in range(1, 421):
        print("Opening As tr image: " + str(number))
        im = Image.open("Dataset/Train/Asian/As_tr" + str(number) + ".jpg")

        resized_image = im.resize((newSize, newSize))
        resized_image.convert('RGB').save("Dataset/Train/Resized_Images/Asian/Asian_" + str(number) + ".jpg")

    for number in range(1, 91):
        print("Opening As te image: " + str(number))
        im = Image.open("Dataset/Train/Asian/As_te" + str(number) + ".jpg")

        resized_image = im.resize((newSize, newSize))
        resized_image.convert('RGB').save("Dataset/Train/Resized_Images/Asian/Asian_" + str(number + 420) + ".jpg")

    print("Images cropped!");


def loadimages():
    print("Loading images ...")
    train_set_x = []
    train_set_y = []

    # Load african images
    for number in range(1, ):
        print("test")

    print("Images loaded!")


if __name__ == '__main__':
    cropimages()
    # loadimages()



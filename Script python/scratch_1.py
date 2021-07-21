import os
import tqdm
import csv
import matplotlib.pyplot as plt
import fnmatch
os.chdir(r'C:\PythonData\LAGGISS')

#from keras import Input, Model
import Class_Image as Ci

class PreprocessingImages:
    """
    Definition of the Preprocessing class.

    This class is used to process different analyses upon all the images of a directory
    and store the results in text files.
    """

    def __init__(self, img_dir):
        self.img_dir = img_dir

        self.blurry_list = []
        self.overexpose_list = []
        self.exposure_histogram_list = []
        self.under_ratio_list = []
        self.winter_list = []
        self.snowy_list = []

        self.good_list = []
        self.bad_list = []

        self.images_list = fnmatch.filter(os.listdir(self.img_dir), '*.jpg')
        self.green_ratio = "Not computed"
def get_filename_from_key(key, image_folder):
    """
    Get the complete path of an image from its key

    :param key: image key
    :type key: str
    :param image_folder: folder containing images
    :type image_folder: str
    :return: image path
    :rtype: str
    """
    images = os.listdir(image_folder)
    for image in images:
        if key in image:
            return os.path.join(image_folder, image)


img_size = 224
image_folder = r'C:\PythonData\LAGGISS\Question_one'
csv_path = r'C:\PythonData\LAGGISS\duels_question_1.csv'
with open(csv_path, 'r') as csvfileReader:
    reader = csv.reader(csvfileReader, delimiter=',')
    print("Creating inputs from csv ...")
    # pbar = progressbar.ProgressBar()
    for line in tqdm(reader, total=5780):
        left_image_path = get_filename_from_key(line[0], image_folder)
        left_img = Ci.Image(left_image_path)
    plt.imshow(left_img)
    plt.show()

#img_a = Input(shape=(img_size, img_size, 3), name="left_image")
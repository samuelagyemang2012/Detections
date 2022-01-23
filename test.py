import os
from bs4 import BeautifulSoup

base = "D:/Datasets/my_coco/test/"
files = os.listdir(base)

for f in files:
    data = open(base + f)
    soup = BeautifulSoup(data, "xml")
    objects = soup.find_all('object')

    for obj in objects:
        # class_name = obj.find('name', recursive=False).text
        obj.find('name', recursive=False).replace_with("car")

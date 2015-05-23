import linecache
import csv
import scipy
import numpy

class DigitClass:
    def __init__(self, digit):
        self.digit = digit
        self.pixels = [[] for i in range(256)]    # pixels[0..255]
        self.means = [-1 for i in range(256)]     # means[0..255]
        self.variances = [-1 for i in range(256)] # variances[0..255]
        self.num_samples = 0

def get_class_list():
    idx_file = open('data/2.csv', 'r')
    idx_reader = csv.reader(idx_file)
    class_list = [DigitClass(digit) for digit in range(10)] # class_list[0..9]
    
    # For each of the student's selected indexes
    for row in idx_reader:
        
        # Readlist corresponding line on usps database
        usps_line = linecache.getline('data/usps.csv', int(row[0]) + 1)
        usps_line = list(csv.reader([usps_line], delimiter=','))[0]
        
        # get class reference
        digit_class = class_list[int(usps_line.pop())]
        digit_class.num_samples += 1
        
        # add each pixel value
        for i in range(256):
            digit_class.pixels[i].append(float(usps_line[i]))
        
    # For each class, calculate each pixel's mean and variance across samples
    for digit_class in class_list:
        for i in range(256):
            digit_class.means[i] = numpy.mean(digit_class.pixels[i])
            digit_class.variances[i] = numpy.var(digit_class.pixels[i])
    
    return class_list

class_list = get_class_list()
for digit_class in class_list:
    print(str(digit_class.digit) + ": " + str(numpy.mean(digit_class.means)))
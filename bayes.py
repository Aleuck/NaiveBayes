from scipy import stats
import linecache
import csv
import numpy



##################################################
## Classes
##################################################

class DigitClass:
    def __init__(self, digit):
        self.digit = digit
        self.pixels = [[] for i in range(256)]     # pixels[0..255][sample]
        self.means = [-1 for i in range(256)]      # means[0..255]
        self.deviations = [-1 for i in range(256)] # deviations[0..255]
        self.num_samples = 0
    
    def get_similarity(self, pixels):
        
        logpdf_sum = 0
        
        for i in range(256):
            logpdf_sum += stats.norm.logpdf(pixels[i], loc=self.means[i], scale=self.deviations[i])
        
        return logpdf_sum



##################################################
## Functions
##################################################

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
            digit_class.deviations[i] = numpy.std(digit_class.pixels[i])
    
    return class_list

def get_sample_pixels(class_list, class_index, sample_index):
    sample_pixels = []
    
    for i in range(256):
        sample_pixels.append(class_list[class_index].pixels[i][sample_index])
    
    return sample_pixels
    
def get_most_likely_digit(class_list, sample_pixels):
    
    most_likely_digit = 0
    max_similarity_value = -float("inf")
    
    for digit_class in class_list:
        
        similarity_value = digit_class.get_similarity(sample_pixels)
        if (similarity_value > max_similarity_value):
            most_likely_digit = digit_class.digit
            max_similarity_value = similarity_value
    
    return most_likely_digit
    


##################################################
## Main
##################################################

class_list = get_class_list()
#for digit_class in class_list:
#    print(str(digit_class.digit) + ": " + str(numpy.mean(digit_class.means)))

samples_per_class = 5 # Change for faster/slower tests

guesses = numpy.zeros((10,10), dtype=numpy.int32)
for class_index in range(10):
    right_guesses = 0
    for sample_index in range(samples_per_class):
        
        sample_pixels = get_sample_pixels(class_list, class_index, sample_index)
        most_likely_digit = get_most_likely_digit(class_list, sample_pixels)
        
        guesses[class_index, most_likely_digit] += 1
        if (most_likely_digit == class_index):
            right_guesses += 1
    
    print(str(class_index) + ": " + str(right_guesses) + "/" + str(samples_per_class))
print("    |  0  1  2  3  4  5  6  7  8  9 | precision    recall  fmeasure")

print("----|-------------------------------|")
for digit in range(10):
    precision = float(guesses[digit,digit]) / float(numpy.sum(guesses[digit,:]))
    recall = float(guesses[digit,digit]) / float(numpy.sum(guesses[:,digit]))
    fmeasure = 0.0
    string = " %2d | " % digit
    for guess in guesses[digit,:]:
        string += "%2d " % guess
    string += "| %1.7f %1.7f %1.7f" % (precision, recall, fmeasure)
    print(string)
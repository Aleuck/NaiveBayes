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
        self.samples = []                          # samples[sample][0..255]
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
        
        # store all samples
        digit_class.samples.append([float(i) for i in usps_line])
        
    return class_list
    
def calculate_class_list_statistics(class_list, first_excluded_sample_id = -1, last_excluded_sample_id = -1):
    
    for digit_class in class_list:
        selected_samples = []
        pixels = [[] for i in range(256)] # pixels[0..255][sample]
        
        # Filter selected samples (don't include excluded samples)
        for sample_id in range(digit_class.num_samples):
            if sample_id < first_excluded_sample_id or sample_id > last_excluded_sample_id:
                selected_samples.append(digit_class.samples[sample_id])
        
        # For each sample, add each of its 256 pixels to each element of pixels[0..255]
        for sample in selected_samples:
            for p in range(256):
                pixels[p].append(sample[p])
        
        # Calculate mean and variance for each pixel across samples
        for p in range(256):
            digit_class.means[p] = numpy.mean(pixels[p])
            digit_class.deviations[p] = numpy.std(pixels[p])
    
def get_most_likely_digit(class_list, sample_pixels):
    
    most_likely_digit = 0
    max_similarity_value = -float("inf")
    
    for digit_class in class_list:
        
        similarity_value = digit_class.get_similarity(sample_pixels)
        if (similarity_value > max_similarity_value):
            most_likely_digit = digit_class.digit
            max_similarity_value = similarity_value
    
    return most_likely_digit

def get_confusion_matrix(class_list, first_test_sample_id, last_test_sample_id):
    
    # Calculate confusion matrix
    guesses = numpy.zeros((10,10), dtype=numpy.int32)
    for class_index in range(10):
        right_guesses = 0
        for sample_index in range(first_test_sample_id, last_test_sample_id + 1):
            
            digit_class = class_list[class_index]
            sample_pixels = digit_class.samples[sample_index]
            most_likely_digit = get_most_likely_digit(class_list, sample_pixels)
            
            guesses[class_index, most_likely_digit] += 1 # Add guess to confusion matrix
            if (most_likely_digit == class_index):
                right_guesses += 1
        
        print(str(class_index) + ": " + str(right_guesses) + "/" + str((last_test_sample_id-first_test_sample_id)+1))
    
    return guesses

def print_results(guesses):
    # Print confusion matrix
    print("    |    0    1    2    3    4    5    6    7    8    9 | precision recall    fmeasure")
    print("----|---------------------------------------------------|")
    for digit in range(10):
        right_guesses = guesses[digit,digit]
        total_guesses = numpy.sum(guesses[:,digit])
        total_occurrences = numpy.sum(guesses[digit,:])
        
        precision = float(right_guesses) / float(total_guesses)
        recall = float(right_guesses) / float(total_occurrences)
        fmeasure = 2.0 * (precision * recall) / (precision + recall)
        
        string = " %2d | " % digit
        for guess in guesses[digit,:]:
            string += "%4.1f " % guess
        string += "| %1.7f %1.7f %1.7f" % (precision, recall, fmeasure)
        print(string)



##################################################
## Main
##################################################

class_list = get_class_list()
confusion_matrices = numpy.zeros((10,10,10));
for partition in range(10):
    partition_size = 50
    first_partition_sample_id = partition_size*(partition)
    last_partition_sample_id  = partition_size*(partition+1) - 1
    
    print("Testando amostras de %d a %d, treinando com amostras restantes:" % (first_partition_sample_id, last_partition_sample_id))
    calculate_class_list_statistics(class_list, first_partition_sample_id, last_partition_sample_id) # Excludes partition's samples from training
    guesses = get_confusion_matrix( class_list, first_partition_sample_id, last_partition_sample_id) # Test with partition's samples
    confusion_matrices[partition,:,:] = guesses
    print_results(guesses)
    
mean_confusion_matrix = numpy.mean(confusion_matrices, axis=0)
print ("Matriz de confusão média:")
print_results(mean_confusion_matrix)
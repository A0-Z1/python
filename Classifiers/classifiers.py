'''
Implement Gaussian Disciminant Analysis and
Naive Bayes Classifier using only built-in
packages 'collections' and 'math'
'''

from collections import defaultdict, Counter
from math import pi, sqrt, exp

class GDA:
    '''
    Implements Gaussian Discriminant Analysis
    '''

    def __init__(self, classes, data):
        '''
        class constructor
        classes is a list of categories,
        data is the correspondent value
        '''

        self.classes = set(classes)
        self.data = data

        # determine category frequencies
        self.freq = self._frequencies(classes)

        # divide data into classes
        self.categories = self._separator(classes)

    def _frequencies(self, classes):
        '''
        Determine category frequencies
        '''
        freq = Counter(classes)
        freq = {class_:value/len(self.data) for class_, value in freq.items()}

        return freq

    def _separator(self, classes):
        '''
        Separates data into classes
        '''

        categories = defaultdict(list)
        for position, class_ in enumerate(classes):
            categories[class_].append(self.data[position])

        return categories

    def fit(self):
        '''
        fit the data. For each class,
        determine sample mean and sample variance
        of training data and store it in a dictionary
        so that we can use it to perform classification.
        For each class there is a list with two elements:
        first element is mean, second is variance
        '''
        self.weights = defaultdict(list)
        for class_ in self.classes:
            data = self.categories[class_]
            self.weights[class_] = [mean(data), var(data)]

    def predict(self, data_point):
        '''
        Predict the class of the datapoint
        by calculating the value of the gaussian
        distribution in 'data_point' for each class,
        then using bayes and picking
        the class with higher probability.
        '''

        scores = {}
        prob = self._probabilities(data_point)
        for class_ in self.classes:
            # determine the values for each class
            p_category = self.freq[class_]

            scores[class_] = bayes(prob[class_], p_category)

        # extract category with max value
        class_, probab = max(scores.items(), key=lambda pair: pair[1])

        return class_, probab

    def _probabilities(self, data_point):
        '''
        Helper method for predict.
        Calculates P(x|y_i)
        '''
        prob= {}
        for class_ in self.classes:
            # determine the values for each class
            mean = self.weights[class_][0]
            var = self.weights[class_][1]
            p_data = self._gauss_distr(data_point, mean, var)
            prob[class_] = p_data

        return prob


    @staticmethod
    def _gauss_distr(x, mean, var):
        '''
        calculate the value of
        the gaussian distribution with mean 'mean'
        and variance 'var' in 'x'
        '''

        coeff = 1 / sqrt(2*pi*var)
        exponent = exp(-(x - mean)**2 / (2*var))

        return coeff * exponent


class NBA:
    '''
    Implement Naive Bayesian Classifier
    '''

    def __init__(self, classes, data):
        '''
        class constructor
        classes is a list of categories,
        data is the correspondent value (list
        of lists or tuples)
        '''

        self.classes = set(classes)
        self.data = data

        # determine category frequencies
        self.freq = self._frequencies(classes)

        # divide data into classes
        self.categories = self._separator(classes)

    def _frequencies(self, classes):
        '''
        Determine category frequencies
        '''
        freq = Counter(classes)
        freq = {class_:value/len(self.data) for class_, value in freq.items()}

        return freq

    def _separator(self, classes):
        '''
        Separates data into classes
        '''

        categories = defaultdict(list)
        for position, class_ in enumerate(classes):
            categories[class_].append(self.data[position])

        return categories

    def fit(self):
        '''
        Create bag of words
        '''
        self.bags = {}
        for categ in self.classes:
            data = self.categories[categ]
            self.bags[categ] = self._bag_of_words(data)

    def predict(self, data_point):
        '''
        Classify data point by first
        calculating prod_i P(data_point | class_i),
        and then applying bayes. Determine
        the class by picking the result with
        highest score
        data_point is a list or tuple
        '''
        # keep scores in dictionary
        scores = {}
        prob = self._probabilities(data_point)
        for categ in self.classes:
            p_category = self.freq[categ]
            scores[categ] = bayes(prob[categ], p_category)

        # extract category with max value
        class_, probab = max(scores.items(), key=lambda pair: pair[1])

        return class_, probab

    def _probabilities(self, data_point):
        '''
        Helper method for predict.
        Calculates prod_i P(x_i|y_k)
        '''
        prob = {}
        for categ in self.classes:
            data = self.bags[categ]
            p_data = 1
            # calculate product between features
            for count in range(len(data_point)):
                feature = data[count]
                # if the value of the feature
                # of data_point wasn't present
                # in training data, return 0
                p_data *= feature.get(data_point[count], 0)

            prob[categ] = p_data

        return prob

    @staticmethod
    def _bag_of_words(data):
        '''
        Create a bag of word:
        for each word contained
        in data, determine the
        frequency according to
        its position and return a dictionary
        '''

        # total length of feature vector
        length = len(data[0])

        # this list contains a dictionary
        # for every feature.
        bags = [defaultdict(int) for i in range(length)]
        # iterate through data
        for el in data:
            for count, token in enumerate(el):
                bags[count][token] += 1

        # determine frequency of each word per category
        for count in range(length):
            bags[count] = {key:value/len(data) for key, value in bags[count].items()}

        return bags


class Combine:
    '''
    We combine gda with nba by simply adopting
    the same approach we had in the naive bayesian
    algorithm: calculate P((x_1, ..., x_n)|y_k) as
    P(x_1|y_k)...P(x_n|y_k). Hower now some of the
    variables are continuous; in that case we assume
    that they have gaussian distribution and calculate
    the probability in the same way we did in gda.
    The rest is the same (apply Bayes rule)
    '''

    def __init__(self, classes, cont_data, discr_data):

        self.classes = set(classes)
        self.cont_data = cont_data
        self.discr_data = discr_data

        self.gda = GDA(classes, cont_data)
        self.nba = NBA(classes, discr_data)

    def fit(self):
        self.gda.fit()
        self.nba.fit()

    def predict(self, cont_data_point, discr_data_point):
        '''
        Predict data. cont_data_point is
        the continuous data, discr_data_point is
        the categorical data
        '''

        # keep scores in dictionary
        scores = {}
        prob = self._probability(cont_data_point, discr_data_point)
        for categ in self.nba.classes:
            p_category = self.gda.freq[categ]
            # apply Bayes
            scores[categ] = bayes(prob[categ], p_category)

        # extract category with max value
        class_, probab = max(scores.items(), key=lambda pair: pair[1])

        return class_, probab

    def _probability(self, cont_data_point, discr_data_point):
        '''
        Helper method for predict. It calculates
        the product between gaussian distributed value
        and bernoulli distributed values
        '''
        prob = {}
        ## determine the values for each class
        # gda part
        cont_p_data = self.gda._probabilities(cont_data_point)
        # nba part
        discr_p_data = self.nba._probabilities(discr_data_point)

        for categ in self.nba.classes:
            # combine the two values
            p_data = cont_p_data[categ] * discr_p_data[categ]
            prob[categ] = p_data

        return prob


def mean(data):
    '''
    calculate the sample mean
    '''

    return sum(data)/len(data)

def var(data):
    '''
    calculate the unbiased sample variance
    '''

    mean_ = mean(data)
    partial = sum((x - mean_)**2 for x in data)

    return partial/(len(data) - 1)

def bayes(p_data, p_category):
    '''
    Use bayes formula (ignoring the term
    at denominator: since given a datapoint,
    the value at denominator is always the same,
    it doesn't have relevance in the comparison)
    '''

    return p_data * p_category

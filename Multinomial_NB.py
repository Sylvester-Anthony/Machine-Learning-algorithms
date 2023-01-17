import numpy as np 
from .BaseClass import BaseClass

class MultinomialNB(BaseClass):
    """
    This code is written to implement Multinomial Naive Bayes from scratch
    
    """
    def __init__(self, k=0.5):
        self.k = k
        self.category0_count = 0
        self.category1_count = 0
        self.total_counts = self.category0_count + self.category1_count
        self.category_0_prior = 0
        self.category_1_prior = 0
        self.category_0_prior, self.category_1_prior
        self.word_probs = []
        self.vocab = []

    def tokenize(self, document):
        """
        To convert a document to a list of words
        
        Args:
        
        documents : a csv file of texts 
        
        Returns:
        
        tokens : list of words
        
        
        """
        doc = document.lower()

        stop_characters = '''0123456789!()-[]{};:'"\,<>./?@#$%^&*_~'''

        tokens = ""

        for character in doc:
          if character not in stop_characters:
            tokens += character

        return tokens.split() 
  
    def count_words(self, X, y):
        """
        Args:
        
        X      :  is an array of documents
        y      :   is an array of target variables
        
        Returns :
        
        counts : counts of the number of words      
        """
        counts = {}
        
        for document, category in zip(X, y):
            for token in self.tokenize(document):
              
              if token not in counts:
                counts[token] = [0,0]
              
              counts[token][category] += 1
        return counts

    def prior_probability(self, counts):
        """
         To loop through the counts dictionary and to get the counts.
         
         Args :
          
          counts : counts of the words
          
         Returns :
         
         category_0_prior, category_1_prior : prior probabilities of category 0 and category 1
         
         
        """
        category0_word_count = category1_word_count = 0
        for word, (category0_count, category1_count) in counts.items():
            category0_word_count += category0_count
            category1_word_count += category1_count

       
        self.category0_count = category0_word_count
        self.category1_count = category1_word_count
        self.total_counts = self.category0_count + self.category1_count

       
        category_0_prior = category0_word_count / self.total_counts
        category_1_prior = category1_word_count / self.total_counts
        return category_0_prior, category_1_prior

    def word_probabilities(self, counts):
        """
        turn the word_counts into a list of probabilities
                Args:
        
        X      :  is an array of documents
        y      :   is an array of target variables
        
        
        """

        self.vocab = [word for word, (category0, category1) in counts.items()]
        return [(word,
        (category0 + self.k) / (self.category0_count + 2 * self.k),
        (category1 + self.k) / (self.category1_count + 2 * self.k))
        for word, (category0, category1) in counts.items()]

    def fit(self, X, y):
        """
        The fit method is used to fit the features X and y.
        Args:
            X      : the array of features
            y      : the array of the variable of interest
        Returns:
            None
        """
        
        counts = self.count_words(X, y)
        self.category_0_prior, self.category_1_prior = self.prior_probability(counts)
        self.word_probs = self.word_probabilities(counts)

    def predict(self, test_corpus):
        """
        The predict method is used to predict the target outcomes for features in the test set
        Args:
            test_corpus   : the array of features for testing
           
        Returns:
            y_pred        : prediction for the test features
        """
      
        y_pred = []
        for document in test_corpus:
          
          log_probabilities_category0 = log_probabilities_category1 = 0.0
          tokens = self.tokenize(document)
           
          for word, probabilities_category0, proabilities_category1 in self.word_probs:
            if word in tokens:
              
              log_probabilities_category0 += np.log(probabilities_category0)
              log_probabilities_category1 += np.log(proabilities_category1)
            
          category_0_pred = self.category_0_prior * np.exp(log_probabilities_category0)
          category_1_pred = self.category_1_prior * np.exp(log_probabilities_category1)
            
          if category_0_pred >= category_1_pred:
            y_pred.append(0)
          else:
            y_pred.append(1)
        return y_pred
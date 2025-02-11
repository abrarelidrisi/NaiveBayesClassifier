import re
import numpy as np
from collections import defaultdict

class NaiveBayesClassifier:
    def __init__(self, smoothing = 1, binarized = False):
        self.smoothing = smoothing
        self.binarized = binarized

        self.vocab = set()
        self.class_priors = {}
        self.class_document_counts = defaultdict(int)
        self.total_words = defaultdict(int)
        self.word_probs = defaultdict(lambda:defaultdict(float))


    def preprocess_text(self, txt):
        txt = txt.lower()
        txt = re.sub(r"[^\w\s]", "", txt)
        return txt.split()
    

    def train(self, documents, labels):

        for words, label in zip(documents, labels):
            self.class_document_counts[label] +=1
            unique_words = set(words) if self.binarized else words

            for word in unique_words:
                self.word_probs[label][word] +=1
                self.vocab.add(word)
                self.total_words[label] +=1 if self.binarized else words.count(word)

        total_docs = len(labels)
        for label in self.class_document_counts:
            self.class_priors[label] = self.class_document_counts[label] / total_docs

        V = len(self.vocab)
        for label in self.class_document_counts:
            for word in self.vocab:
                self.word_probs[label][word] = (
                  (self.word_probs[label][word] + self.smoothing) / (self.total_words[label] + V)
                )

    def predict(self, text):
        words = self.preprocess_text(text)
        scores = {}

        print(f"\nProcessing sentence: \"{text}\"")
        print(f"Tokenized words: {words}\n")


        for label in self.class_document_counts:
            score = np.log(self.class_priors[label])  
            unique_words = set(words) if self.binarized else words

            print(f"Computing probability for class: {label}")
            print(f"Prior log probability: {np.log(self.class_priors[label])}")

            for word in unique_words:
                if word in self.word_probs[label]:
                    log_probability = np.log(self.word_probs[label][word])
                    score += log_probability
                    print(f"Word '{word}': P({word}|{label} = {self.word_probs[label][word]}, log_prob = {log_probability}")

            scores[label] = score
            print(f"Final log probability for {label}: {score}\n")

        predicted_label = max(scores, key=scores.get)
        print(f"Final prediction: {predicted_label.upper()} ✅\n")
        return predicted_label
    
n = int(input("Number of examples:"))
documents = []
labels = []
for i in range(n):
    sentence = input("Please provide a sentence:").lower()
    l = input("Give corresponding label (pos/neg):")
    sentence = re.sub(r"[^\w\s]", "", sentence)
    sentence = sentence.split()
    documents.append(sentence)
    labels.append(l)


mnb = NaiveBayesClassifier(binarized=False)
bnb = NaiveBayesClassifier(binarized=True)

mnb.train(documents, labels)
bnb.train(documents, labels)

test_sentence = "A good, good plot and great characters, but poor acting."

print("\n--- Multinomial Naïve Bayes ---")
mnb_prediction = mnb.predict(test_sentence)

print("\n--- Binarized Naïve Bayes ---")
bnb_prediction = bnb.predict(test_sentence)

print(documents)

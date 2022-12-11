from retriever import *
from qa import *
from evaluation import *
from dataset import *
from transformers import BertForQuestionAnswering, BertTokenizer
import numpy as np

import argparse
 
 
# Initialize parser
parser = argparse.ArgumentParser("main.py")
 
# Adding optional argument
parser.add_argument("-o", "--Output", help = "Show Output")
 
# Read arguments from command line
args = parser.parse_args()


if __name__ == "__main__":
    print("Loading Dataset")
    url = 'https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json'
    passages, questions, answers = load_dataset(url)
    print("========================\n")

    # load the models
    print("Loading models")
    tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
    reader = BertForQuestionAnswering.from_pretrained('bert-base-uncased')
    retriever = LexicalSearch()
    retriever.fit(passages)
    print("========================\n")

    # randomly select 1000 questions for eval
    np.random.seed(5300)
    idx = np.random.choice(np.arange(len(questions)), 1000)
    sample_questions = np.array(questions)[idx]
    sample_answers = [ans['text'] for ans in np.array(answers)[idx]]

    # evalute the qa system
    print("Evaluating\n")
    qa = QuestionAnswering(tokenizer, retriever, reader)
    em, f1_score = eval_reader(qa, sample_questions, sample_answers)


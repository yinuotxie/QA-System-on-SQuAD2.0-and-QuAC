from retriever import *
from qa import *
from evaluation import *
from dataset import *
from transformers import BertForQuestionAnswering, BertTokenizer
from sentence_transformers import SentenceTransformer, CrossEncoder, util
import numpy as np


if __name__ == "__main__":
    print("Loading Dataset")
    url = 'https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json'
    passages, questions, answers = load_dataset(url)
    print("========================\n")
    
    # load the models
    print("Loading models")
    bi_encoder = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')           
    bi_encoder.max_seq_length=512             
    cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
    reader = BertForQuestionAnswering.from_pretrained('bert-base-uncased')
    retriever = PassageRanking(bi_encoder, cross_encoder)
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


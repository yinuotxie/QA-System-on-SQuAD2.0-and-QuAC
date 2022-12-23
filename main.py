from retriever import *
from qa import *
from evaluation import *
from dataset import *
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from sentence_transformers import SentenceTransformer, CrossEncoder
import argparse
import numpy as np


# Adding optional argument
def _parse_args():
    parser = argparse.ArgumentParser(description='QpenDomainQA System')

    # General system running and configuration options
    parser.add_argument('--dataset', type=str, choices=['squad', 'quac'], default='squad',
                        help='which dataset to use to eval')
    parser.add_argument('--model_name', type=str, default='bert-large-uncased-whole-word-masking-finetuned-squad',
                        help='model name')
    parser.add_argument('--retriever', type=str, choices=['lexical', 'semantic'], default='lexical',
                        help='methods of retriever')
    parser.add_argument('--apply_retriever', default=True, action='store_false', help='whether to use retriever')
    parser.add_argument('--baseline', default=False, action='store_true', help='run the simple baseline')
    parser.add_argument('--print_examples', default='True', action='store_false', help="Print examples")
    parser.add_argument('--eval', default=False, action='store_true', help='whether to eval the pipeline.')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = _parse_args()
    # assert args.
    print("Loading Dataset")
    squad_train_context, squad_train_queries, squad_train_answers = load_dataset(
        'https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json')
    squad_dev_context, squad_dev_queries, squad_dev_answers = load_dataset(
        'https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json')
    quac_train_context, quac_train_queris, quac_train_answers = load_dataset(
        'https://s3.amazonaws.com/my89public/quac/train_v0.2.json')
    quac_dev_context, quac_dev_queries, quac_dev_answers = load_dataset(
        'https://s3.amazonaws.com/my89public/quac/val_v0.2.json')
    print("Length of SQuAD2.0 Train Dataset:", len(squad_train_context))
    print("Length of SQuAD2.0 Dev Dataset:", len(squad_dev_context))
    print("Length of SQuAD2.0 Train Dataset:", len(squad_train_context))
    print("Length of SQuAD2.0 Dev Dataset:", len(squad_dev_context))
    print("========================\n")

    if args.apply_retriever:
        print("Load Retriever")
        print("Retriever Method:", args.retriever)
        if args.retriever == 'lexical':
            retriever = LexicalSearch()
        else:
            bi_encoder = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')
            # bi_encoder = SentenceTransformer('pinecone/bert-retriever-squad2')
            bi_encoder.max_seq_length = 512
            cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
            retriever = PassageRanking(bi_encoder, cross_encoder)

        retriever.fit(squad_train_context + squad_dev_context + quac_train_context + quac_dev_context)
    else:
        retriever = None
        print("None retriever is used")

    # load the models
    print("Loading Models")
    print("Model Name:", args.model_name)
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

    reader = AutoModelForQuestionAnswering.from_pretrained(args.model_name)
    qa = QuestionAnswering(tokenizer, reader, retriever=retriever, baseline=args.baseline)

    print("========================\n")

    if args.eval:
        # evalute the qa system
        print("Evaluating")
        if args.dataset == 'squad':
            em, f1_score = eval_model(qa, squad_dev_queries, squad_dev_context, squad_dev_answers,
                                      apply_retriever=args.apply_retriever, display=False)
        else:
            em, f1_score = eval_model(qa, quac_dev_queries, quac_dev_context, quac_dev_answers,
                                      apply_retriever=args.apply_retriever, display=False)

        print(f'Exact match: {em}')
        print(f'F1 score: {f1_score}\n')

    if args.print_examples:
        # generate ten examples from the dev dataset
        dev_questions = squad_dev_queries + quac_dev_queries
        dev_context = squad_dev_context + quac_dev_context
        dev_answers = squad_dev_answers + quac_dev_answers

        np.random.seed(5300)
        idx = np.random.choice(np.arange(len(dev_questions)), 10)
        sample_questions = np.array(dev_questions)[idx]
        sample_context = np.array(dev_context)[idx]
        sample_answers = [[text['text'] for text in ans] for ans in np.array(dev_answers, dtype=object)[idx]]

        for question, context, answer in zip(sample_questions, sample_context, sample_answers):
            print(f"Question: {question}")
            if args.apply_retriever:
                prediction = qa.question_answer(question, display=True)
            else:
                prediction = qa.question_answer(question, context, display=True)

            em_score = False
            f1_score = float('-inf')
            if len(answer) == 0:
                true_answer = ''
                em_score = em_score or exact_match(prediction, true_answer)
                f1_score = max(f1_score, compute_f1(prediction, true_answer))
            else:
                for true_answer in answer:
                    em_score = em_score or exact_match(prediction, true_answer)
                    f1_score = max(f1_score, compute_f1(prediction, true_answer))

            print(f'Prediction: {prediction}')
            print(f'True Answer: {answer}\n')
            print(f'Exact match: {em_score}')
            print(f'F1 score: {f1_score}\n')


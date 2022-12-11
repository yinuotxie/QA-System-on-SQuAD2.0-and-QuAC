"""
Evaluation file for Document Retriever and Document Reader
"""
import string
import re


def eval_retriever(retriever, context, queries):
    correct = 0
    for i, q in enumerate(queries):
        for pred in retriever.search(q, top_k=5):
            if pred == context[i]:
                correct += 1
                break

        if (i + 1) % 1000 == 0:
            print(f"Accuracy for the first {i + 1} Query: {correct / (i + 1)}")

    return correct / len(queries)


def normalize_text(s):
    """Removing articles and punctuation, and standardizing whitespace are all typical text processing steps."""

    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def exact_match(prediction, truth):
    return bool(normalize_text(prediction) == normalize_text(truth))


def compute_f1(prediction, truth):
    pred_tokens = normalize_text(prediction).split()
    truth_tokens = normalize_text(truth).split()

    # if either the prediction or the truth is no-answer then f1 = 1 if they agree, 0 otherwise
    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return int(pred_tokens == truth_tokens)

    common_tokens = set(pred_tokens) & set(truth_tokens)

    # if there are no common tokens then f1 = 0
    if len(common_tokens) == 0:
        return 0

    prec = len(common_tokens) / len(pred_tokens)
    rec = len(common_tokens) / len(truth_tokens)

    return round(2 * (prec * rec) / (prec + rec), 2)


def eval_reader(model, questions, answers, display=False):
    total_em = 0
    total_f1 = 0
    size = len(questions)
    batch = size // 10

    for i, (question, answer) in enumerate(zip(questions, answers)):
        prediction = model.question_answer(question, display)
        em_score = False
        f1_score = float('-inf')
        if len(answer) == 0:
            true_answer = ''
            em_score = em_score or exact_match(prediction, true_answer)
            f1_score = max(f1_score, compute_f1(prediction, true_answer))
        else:
            for ans in answer:
                true_answer = ans['text']
                em_score = em_score or exact_match(prediction, true_answer)
                f1_score = max(f1_score, compute_f1(prediction, true_answer))

        total_f1 += f1_score
        if em_score:
            total_em += 1

        if (i + 1) % batch == 0:
            print(f"Exact Match Rate for {(i + 1)}/{size}  Questions: {total_em / (i + 1)}")
            print(f"F1-score for {(i + 1)}/{size}  Questions: {total_f1 / (i + 1)}\n")

    return total_em / size, total_f1 / size

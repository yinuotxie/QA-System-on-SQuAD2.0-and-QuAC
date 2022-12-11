"""
Evaluation file for Document Retriever and Document Reader
"""
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
    import string, re
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
    em = 0
    f1 = 0
    for i, (question, answer) in enumerate(zip(questions, answers)):
        prediction = model.question_answer(question, display)
        em_score = exact_match(prediction, answer)
        f1_score = compute_f1(prediction, answer)
        f1 += f1_score

        if display:
            print(f"Question: {question}")
            print(f'Prediction: {prediction}')
            print(f'True Answer: {answer}\n')
            print(f'Exact match: {em_score}')
            print(f'F1 score: {f1_score}\n')

        if em_score:
            em += 1

        if (i + 1) % 100 == 0:
            print(f"Exact Match Rate for first {i + 1} Questions: {em / (i + 1)}")
            print(f"F1-score for first {i + 1} Questions: {f1 / (i + 1)}\n")

    return em / len(questions), f1_score / len(questions)

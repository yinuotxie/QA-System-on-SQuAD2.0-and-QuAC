# OpenDomainQA Scoring
CIS 5300 (Computational Linguistics) Final Project Scoring method

score = count of retrieved passages contain the real passage/count of all questions

To evaluate the reader we use the standard SQuAD2.0 performance metrics:
Exact Match (EM) score and F1 score. For our project, we focus on the EM
and F1 scores with respect to the dev set. 

• Exact Match: A binary measure of whether the system output matches
the ground truth answer exactly.

• F1: Harmonic mean of precision and recall, where precision = (true posi-
tives) / (true positives + false positives) and recall = true positives / (false
negatives + true positives). F1 score = (2×prediction×recall) / (precision + recall).

To execute the evaluation, the below command can be used,
python <filpath to main.py> --eval

#### To calculate scores mentioned as above, you can call eval_reader(model, questions, answers, display) and set display = True if you want the result printed 

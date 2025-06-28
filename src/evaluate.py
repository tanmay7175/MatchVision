from sklearn.metrics import accuracy_score

def evaluate(predictions, ground_truth):
    y_pred = [x[1] for x in predictions]
    y_true = [x[0] for x in predictions]
    return accuracy_score(y_true, y_pred)
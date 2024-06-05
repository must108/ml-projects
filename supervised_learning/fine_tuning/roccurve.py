
# the roc curve shows the true positive/false positive rates
# at different thresholds

# check 'image.png' for a visual

from sklearn.metrics import roc_curve
from matplotlib.pyplot import plt
fpr, tpr, thresholds = roc_curve(y_test, y_pred_probs)

plt.plot([0, 1], [0, 1], "k--")
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Logistic Regression ROC Curve')

plt.show()

# 1 = true positive
# 0 = false positive

# calculate the area under the ROC curve to find the efficiency of the model
# do this with:

from sklearn.metrics import roc_auc_score
roc_auc_score(y_test, y_pred_probs)
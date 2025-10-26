import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc, confusion_matrix
import matplotlib.pyplot as plt
from collections import Counter

''' Multivariate Gaussian Bayesian classifier '''
def Multivariate_Gaussian_Distribution_likelihood(x, mu, cov):
    d = len(x)
    cov = cov + np.eye(d)  
    diff = x - mu
    likelihood = np.exp(-0.5 * diff @ np.linalg.inv(cov) @ diff) / np.sqrt((2 * np.pi)**d * np.linalg.det(cov))
    return likelihood

def Bayesian_decision_classifier(x, mu0, mu1, cov0, cov1, p0, p1):
    l0 = Multivariate_Gaussian_Distribution_likelihood(x, mu0, cov0)
    l1 = Multivariate_Gaussian_Distribution_likelihood(x, mu1, cov1)
    evidence = (l0 * p0) + (l1 * p1)
    posterior_0 = (l0 * p0) / evidence
    posterior_1 = (l1 * p1) / evidence

    # Discriminant functions
    g1_x = np.log(l1) + np.log(p1) 
    g0_x = np.log(l0) + np.log(p0)
    delta = g1_x - g0_x

    return delta, posterior_1    

''' BIC score (Bayesian Information Criterion)'''
def BIC_score(X_train, y_train):
    classes = np.unique(y_train)
    log_like = 0.0
    n = len(y_train)
    k = 0
    for c in classes:
        X_c = X_train[y_train == c]
        mu = np.mean(X_c, axis=0)
        cov = np.atleast_2d(np.cov(X_c, rowvar=False))
        for x in X_c:
            l = Multivariate_Gaussian_Distribution_likelihood(x, mu, cov, ridge=1e-6)
            log_like = log_like + np.log(l)
        k = k + len(mu) + (len(mu)*(len(mu)+1))/2
    bic = np.log(n) * k - 2 * log_like # smaller is better
    return bic

''' Forward Feature Selection based on BIC '''
def Forward_Selection(X_train, y_train, max_features, min_features):
    remaining = list(range(X_train.shape[1]))
    selected = []
    best_score = np.inf

    while remaining and len(selected) < max_features:
        scores = []
        for f in remaining:
            trial = selected + [f]
            score = BIC_score(X_train[:, trial], y_train)
            scores.append((score, f))
            #print(f"  Trial features: {[feature_names[j] for j in trial]}, BIC score: {score:.2f}")

        new_score, new_f = min(scores, key=lambda x: x[0])  # 注意這裡要取最小的分數（BIC越小越好）
        if new_score < best_score or len(selected) < min_features:
            best_score = new_score
            selected.append(new_f)
            remaining.remove(new_f)
        else:
            break
    return selected

''' Load Data '''
data = pd.read_excel('AcromegalyFeatureSet.xlsx')
data.rename(columns=lambda s: s.strip() if isinstance(s, str) else s, inplace=True)
X = data.drop(columns=['SeqNum','Gender', 'GroundTruth']).values
y = data['GroundTruth'].values
feature_names = data.drop(columns=['SeqNum','Gender', 'GroundTruth']).columns
n = len(y)

''' Leave-One-Out Cross Validation '''
posts1 = []
selected_features = []
deltas = []
print(f"Leave-One-Out Cross Validation")
for i in range(n):
    X_train = np.delete(X, i, axis=0)
    y_train = np.delete(y, i)
    X_test = X[i]

    n0 = int(np.sum(y_train == 0))
    n1 = int(np.sum(y_train == 1))
    print(f"-----------------------------------------------------------")
    print(f"Fold: {i+1}/{n}, Class 0 samples: {n0}, Class 1 samples: {n1}")

    max_features = min(max(1, min(n0, n1) - 1), X_train.shape[1]) 

    selected = Forward_Selection(X_train, y_train, max_features, 3)
    selected_features.append([feature_names[j] for j in selected])
    print(f"Selected features: {[feature_names[j] for j in selected]}")
    X_train_sel = X_train[:, selected]
    X_test_sel = X_test[selected]

    # mu0, mu1 = mean vectors for class 0 and 1
    mu0 = X_train_sel[y_train == 0].mean(axis=0)
    mu1 = X_train_sel[y_train == 1].mean(axis=0)

    # cov0, cov1 = covariance matrices for class 0 and 1
    cov0 = np.atleast_2d(np.cov(X_train_sel[y_train == 0], rowvar=False))
    cov1 = np.atleast_2d(np.cov(X_train_sel[y_train == 1], rowvar=False))

    # p0 = prior for class 0, p1 = prior for class 1
    p0, p1 = np.mean(y_train == 0), np.mean(y_train == 1)

    delta, posterior1 = Bayesian_decision_classifier(X_test_sel, mu0, mu1, cov0, cov1, p0, p1)
    posts1.append(posterior1)
    deltas.append(delta)
posts1 = np.array(posts1)
deltas = np.array(deltas)

''' Performance '''
fpr, tpr, _ = roc_curve(y, posts1)
roc_auc = auc(fpr, tpr)
preds1 = (deltas >= 0).astype(int)
cm = confusion_matrix(y, preds1)
TN, FP, FN, TP = cm.ravel()
acc = (TP + TN) / np.sum(cm)
sen = TP / (TP + FN)
spe = TN / (TN + FP)
print(f"-----------------------------------------------------------")
print("Test Performance")
print(f"Confusion Matrix:\n{cm}")
print(f"Accuracy={acc:.3f}, Sensitivity={sen:.3f}, Specificity={spe:.3f}, AUC={roc_auc:.3f}")

# ROC curve plot
plt.plot(fpr, tpr, label=f"AUC={roc_auc:.2f}")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.title('ROC Curve')
plt.savefig('roc_curve.png')

''' Bivariate Gaussian Bayes classifier with top-2 features '''
counts = Counter([f for fs in selected_features for f in fs])
print('Top-2 features:', counts.most_common(2))

Top2_features = [f for f, _ in counts.most_common(2)]

Top2_X = data[Top2_features].values.astype(float)
Top2_y = data['GroundTruth'].values.astype(int)

Top2_X0, Top2_X1 = Top2_X[Top2_y==0], Top2_X[Top2_y==1]
Top2_mu0, Top2_mu1 = Top2_X0.mean(0), Top2_X1.mean(0)
Top2_cov0, Top2_cov1 = np.cov(Top2_X0,rowvar=False), np.cov(Top2_X1,rowvar=False)
Top2_p0, Top2_p1 = len(Top2_X0)/len(Top2_X), len(Top2_X1)/len(Top2_X)

# grid points
m, M = Top2_X.min(0), Top2_X.max(0); pad = 0.05*(M-m)
xs = np.linspace(m[0]-pad[0], M[0]+pad[0], 300)
ys = np.linspace(m[1]-pad[1], M[1]+pad[1], 300)
xx, yy = np.meshgrid(xs, ys)
pts = np.c_[xx.ravel(), yy.ravel()]

# log-likelihoods and decision boundary
LL0 = np.array(np.log([Multivariate_Gaussian_Distribution_likelihood(p, Top2_mu0, Top2_cov0) for p in pts])).reshape(xx.shape)
LL1 = np.array(np.log([Multivariate_Gaussian_Distribution_likelihood(p, Top2_mu1, Top2_cov1) for p in pts])).reshape(xx.shape)
g0 = LL0 + np.log(Top2_p0)
g1 = LL1 + np.log(Top2_p1)
Top2_delta = g1 - g0  # decision boundary: delta=0

# plot
plt.figure(figsize=(7,6))
plt.scatter(Top2_X0[:,0], Top2_X0[:,1], marker='o', label='Class 0', color = 'gray')
plt.scatter(Top2_X1[:,0], Top2_X1[:,1], marker='+', label='Class 1', color = 'green')
levels0 = np.unique(np.sort(np.percentile(LL0, [20, 40, 60, 80])))
levels1 = np.unique(np.sort(np.percentile(LL1, [20, 40, 60, 80])))

plt.contour(xx, yy, LL0, levels=levels0, linestyles='dashed')
plt.contour(xx, yy, LL1, levels=levels1, linestyles='solid')

# The decision boundary
plt.contour(xx, yy, Top2_delta, levels=[0.0], linewidths=2, colors='black')

plt.xlabel(Top2_features[0]); plt.ylabel(Top2_features[1])
plt.title('Bivariate Gaussian Bayes (Top-2 features)')
plt.legend(); plt.tight_layout()
plt.savefig('bivariate_bayes_qda.png', dpi=180)






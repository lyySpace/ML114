import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc, confusion_matrix
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

''' Multivariate Gaussian Bayesian classifier '''
def Multivariate_Gaussian_Distribution_log_likelihood(X, mu, cov):
    X = np.atleast_2d(X)               
    d = X.shape[1]
    L = np.linalg.cholesky(cov)
    Z = np.linalg.solve(L, (X - mu).T)  
    quad = np.sum(Z*Z, axis=0)          
    logdet = 2.0 * np.sum(np.log(np.diag(L)))
    log_likelihood = -0.5 * (d*np.log(2*np.pi) + logdet + quad)  
    return log_likelihood

def Bayesian_decision_classifier(X, mu0, mu1, cov0, cov1, p0, p1, eps=1e-15):
    ll0 = Multivariate_Gaussian_Distribution_log_likelihood(X, mu0, cov0) + np.log(p0 + eps)  # (n,)
    ll1 = Multivariate_Gaussian_Distribution_log_likelihood(X, mu1, cov1) + np.log(p1 + eps)  # (n,)
    delta = ll1 - ll0
    m = np.maximum(ll0, ll1)
    num1 = np.exp(ll1 - m)
    den  = np.exp(ll0 - m) + num1
    posterior_1 = num1 / den
    # If single sample, return scalars
    if posterior_1.shape[0] == 1:
        return float(delta[0]), float(posterior_1[0])
    return delta, posterior_1

''' MLE Estimator for Gaussian parameters '''
def MLE_Estimater(X, y, ridge=1e-6):
    X0, X1 = X[y == 0], X[y == 1]
    mu0, mu1 = X0.mean(0), X1.mean(0)
    cov0 = np.atleast_2d(np.cov(X0, rowvar=False)) + np.eye(X.shape[1]) * ridge
    cov1 = np.atleast_2d(np.cov(X1, rowvar=False)) + np.eye(X.shape[1]) * ridge
    p0, p1 = len(X0)/len(X), len(X1)/len(X)
    return mu0, mu1, cov0, cov1, p0, p1

''' Cross-Validation AUC Score '''
def cv_AUC_score(X_train_sub, y_train_sub, max_splits=5, random_state=42):
    n_pos = int(np.sum(y_train_sub == 1))
    n_neg = int(np.sum(y_train_sub == 0))
    n_splits = max(2, min(max_splits, n_pos, n_neg))
    if n_pos == 0 or n_neg == 0 or n_splits < 2:
        return 0.5  

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    y_all, s_all = [], []
    for tr_idx, va_idx in skf.split(X_train_sub, y_train_sub):
        Xtr, Xva = X_train_sub[tr_idx], X_train_sub[va_idx]
        ytr, yva = y_train_sub[tr_idx], y_train_sub[va_idx]
        mu0, mu1, cov0, cov1, p0, p1 = MLE_Estimater(Xtr, ytr, ridge=1e-6)
        _, s = Bayesian_decision_classifier(Xva, mu0, mu1, cov0, cov1, p0, p1)  
        y_all.append(yva); s_all.append(s)
    y_all = np.concatenate(y_all); s_all = np.concatenate(s_all)
    try:
        return roc_auc_score(y_all, s_all)
    except Exception:
        return 0.5

''' Forward Feature Selection based on CV-AUC '''
def Forward_Selection_AUC(X_train, y_train, max_features):
    remaining = list(range(X_train.shape[1]))
    selected = []
    best_score = -np.inf  

    while remaining and len(selected) < max_features:
        candidates = []
        for f in remaining:
            trial = selected + [f]
            score = cv_AUC_score(X_train[:, trial], y_train)
            candidates.append((score, f))
        new_score, new_f = max(candidates, key=lambda x: x[0])  

        if  (new_score > best_score + 1e-9):
            best_score = new_score
            selected.append(new_f)
            remaining.remove(new_f)
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

    selected = Forward_Selection_AUC(X_train, y_train, max_features)

    selected_features.append([feature_names[j] for j in selected])
    print(f"Selected features: {[feature_names[j] for j in selected]}")
    X_train_sel = X_train[:, selected]
    X_test_sel = X_test[selected]

    # mu0, mu1 = mean vectors for class 0 and 1
    mu0 = X_train_sel[y_train == 0].mean(axis=0)
    mu1 = X_train_sel[y_train == 1].mean(axis=0)

    # cov0, cov1 = covariance matrices for class 0 and 1
    cov0 = np.atleast_2d(np.cov(X_train_sel[y_train == 0], rowvar=False)) + np.eye(X_train_sel.shape[1]) * 1e-6
    cov1 = np.atleast_2d(np.cov(X_train_sel[y_train == 1], rowvar=False)) + np.eye(X_train_sel.shape[1]) * 1e-6

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
Top2_cov0 = np.cov(Top2_X0, rowvar=False) + np.eye(Top2_X0.shape[1]) * 1e-6
Top2_cov1 = np.cov(Top2_X1, rowvar=False) + np.eye(Top2_X1.shape[1]) * 1e-6
Top2_p0, Top2_p1 = len(Top2_X0)/len(Top2_X), len(Top2_X1)/len(Top2_X)

# grid points
m, M = Top2_X.min(0), Top2_X.max(0); pad = 0.05*(M-m)
xs = np.linspace(m[0]-pad[0], M[0]+pad[0], 300)
ys = np.linspace(m[1]-pad[1], M[1]+pad[1], 300)
xx, yy = np.meshgrid(xs, ys)
pts = np.c_[xx.ravel(), yy.ravel()]

# log-likelihoods and decision boundary
LL0 = Multivariate_Gaussian_Distribution_log_likelihood(pts, Top2_mu0, Top2_cov0).reshape(xx.shape)
LL1 = Multivariate_Gaussian_Distribution_log_likelihood(pts, Top2_mu1, Top2_cov1).reshape(xx.shape)
g0 = LL0 + np.log(Top2_p0)
g1 = LL1 + np.log(Top2_p1)
Top2_delta = g1 - g0 # decision boundary: delta=0

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
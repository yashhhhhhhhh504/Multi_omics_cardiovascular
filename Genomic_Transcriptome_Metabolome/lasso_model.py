
# !pip install --quiet GEOparse pandas numpy scikit-learn matplotlib seaborn

import os
import warnings
import numpy as np
import pandas as pd
import GEOparse
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import (
    train_test_split,
    RepeatedStratifiedKFold,
    cross_validate
)
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix,
    roc_curve, auc
)
from sklearn.exceptions import ConvergenceWarning
def load_transcriptome():
    SOFT = "GSE57338_family.soft"
    if not os.path.exists(SOFT):
        os.system(
            f"wget -q -O {SOFT}.gz "
            "ftp://ftp.ncbi.nlm.nih.gov/geo/series/GSE57nnn/GSE57338/soft/GSE57338_family.soft.gz"
        )
        os.system(f"gunzip -f {SOFT}.gz")
    gse = GEOparse.get_GEO(filepath=SOFT, annotate_gpl=False)
    expr = gse.pivot_samples("VALUE").T
    titles = gse.phenotype_data["title"]
    y = (~titles.str.contains("Non-failing", case=False)).astype(int).values
    print(f"[Transcriptome] shape={expr.shape}, labels={np.bincount(y)}")
    return expr, y
def load_genomics(path="cad.add.160614.website.txt"):
    df = pd.read_csv(path, sep=r"\s+", engine="python",
                     usecols=["markername","beta","p_dgc"])
    df["p_dgc"] = df["p_dgc"].replace(0, 1e-300)
    df["logp"]  = -np.log10(df["p_dgc"])
    top = df.nlargest(100, "logp").reset_index(drop=True)
    print(f"[Genomics] selected top {len(top)} SNPs")
    return top
def load_metabolomics(path="ukb_nightingale_biomarker_atlas.csv"):
    df = pd.read_csv(path)
    sel = df[(df.icd10=="I25") & (df.endpoint_type=="incident")].copy()
    sel["pvalue"] = sel["pvalue"].clip(lower=1e-300)
    sel["logp"]    = -np.log10(sel["pvalue"])
    topm = sel.nlargest(100, "logp").reset_index(drop=True)
    print(f"[Metabolomics] selected top {len(topm)} metabolites")
    return topm
def main():
    # 4.1 Transcriptome: load, split, impute+scale, select features
    expr_df, y = load_transcriptome()
    X_tr, X_te, y_tr, y_te = train_test_split(
        expr_df, y, test_size=0.2, stratify=y, random_state=42
    )
    imp = SimpleImputer(strategy="median")
    sc  = StandardScaler()
    X_tr = sc.fit_transform(imp.fit_transform(X_tr))
    X_te = sc.transform(imp.transform(X_te))
    lasso = LogisticRegressionCV(
        penalty='l1', solver='saga',
        cv=5, scoring='roc_auc',
        max_iter=10000, tol=1e-4,
        verbose=0, n_jobs=-1
    )
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=ConvergenceWarning)
        lasso.fit(X_tr, y_tr)
    sel = SelectFromModel(lasso, max_features=50, prefit=True)
    X_tr_sel = sel.transform(X_tr)
    X_te_sel = sel.transform(X_te)
    genes_sel = expr_df.columns[sel.get_support()]
    print(f"[LASSO] selected {X_tr_sel.shape[1]} transcriptomic features")
    top_snp   = load_genomics()
    G_tr      = np.tile(top_snp.beta.values, (X_tr_sel.shape[0],1))
    G_te      = np.tile(top_snp.beta.values, (X_te_sel.shape[0],1))
    gsc       = StandardScaler().fit(G_tr)
    G_tr      = gsc.transform(G_tr)
    G_te      = gsc.transform(G_te)
    snps_sel  = top_snp.markername.tolist()
    top_met  = load_metabolomics()
    M_tr     = np.tile(top_met.logp.values, (X_tr_sel.shape[0],1))
    M_te     = np.tile(top_met.logp.values, (X_te_sel.shape[0],1))
    msc      = StandardScaler().fit(M_tr)
    M_tr     = msc.transform(M_tr)
    M_te     = msc.transform(M_te)
    mets_sel = top_met.biomarker_id.tolist()
    X_tr_all = np.hstack([X_tr_sel, G_tr, M_tr])
    X_te_all = np.hstack([X_te_sel, G_te, M_te])
    print(f"[Fuse] train={X_tr_all.shape}, test={X_te_all.shape}")
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=5,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    )
    rskf = RepeatedStratifiedKFold(
        n_splits=5, n_repeats=5, random_state=42
    )
    scoring = ['accuracy','precision','recall','f1','roc_auc']
    cv_res = cross_validate(
        rf, X_tr_all, y_tr,
        cv=rskf, scoring=scoring, n_jobs=-1,
        return_train_score=False
    )
    print("\n[CV Results]")
    for m in scoring:
        mean = cv_res[f'test_{m}'].mean()
        std  = cv_res[f'test_{m}'].std()
        print(f"  {m:>8}: {mean:.3f} ± {std:.3f}")
    rf.fit(X_tr_all, y_tr)
    y_pred = rf.predict(X_te_all)
    y_prob = rf.predict_proba(X_te_all)[:,1]
    print("\n[Test Metrics]")
    print(f"  Accuracy : {accuracy_score(y_te, y_pred):.3f}")
    print(f"  Precision: {precision_score(y_te, y_pred):.3f}")
    print(f"  Recall   : {recall_score(y_te, y_pred):.3f}")
    print(f"  F1-score : {f1_score(y_te, y_pred):.3f}")
    print("\n" + classification_report(y_te, y_pred))
    cm = confusion_matrix(y_te, y_pred)
    plt.figure(figsize=(4,3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Non-HF","HF"], yticklabels=["Non-HF","HF"])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.tight_layout(); plt.savefig("confusion_matrix.png"); plt.close()
    fpr, tpr, _ = roc_curve(y_te, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(5,4))
    plt.plot(fpr, tpr, lw=2, label=f"AUC={roc_auc:.2f}")
    plt.plot([0,1],[0,1],'--',color='gray')
    plt.title("Test ROC Curve"); plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.legend(loc="lower right")
    plt.tight_layout(); plt.savefig("roc_multiomic.png"); plt.close()
    imp_vals = rf.feature_importances_
    t, g, m = X_tr_sel.shape[1], G_tr.shape[1], M_tr.shape[1]

    def plot_top(arr, names, fname, title):
        idx = np.argsort(arr)[-10:][::-1]
        fig, ax = plt.subplots(figsize=(6,4))
        colors = plt.cm.viridis(np.linspace(0,1,10))
        ax.barh(np.arange(10), arr[idx], color=colors)
        ax.set_yticks(np.arange(10))
        ax.set_yticklabels(np.array(names)[idx], fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("Importance"); ax.set_title(title)
        plt.tight_layout(); plt.savefig(fname); plt.close()

    plot_top(imp_vals[:t], genes_sel,   "feat_genes.png",   "Top 10 Transcriptomic")
    plot_top(imp_vals[t:t+g], snps_sel, "feat_snps.png",    "Top 10 Genomic")
    plot_top(imp_vals[t+g:], mets_sel,  "feat_mets.png",    "Top 10 Metabolomic")

    print("\nPlots saved:")
    print(" - confusion_matrix.png")
    print(" - roc_multiomic.png")
    print(" - feat_genes.png, feat_snps.png, feat_mets.png")
if __name__ == "__main__":
    main()

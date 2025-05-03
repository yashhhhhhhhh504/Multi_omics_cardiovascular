#!/usr/bin/env python3
# multiomic_model_comparison.py
import os, warnings
import numpy as np
import pandas as pd
import GEOparse
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix,
    roc_curve, auc
)
from sklearn.exceptions import ConvergenceWarning
# ─── 1) TRANSCRIPTOME ───
def load_transcriptome():
    SOFT = "GSE57338_family.soft"
    if not os.path.exists(SOFT):
        print("Downloading GSE57338 …")
        os.system(
            f"wget -q -O {SOFT}.gz "
            "ftp://ftp.ncbi.nlm.nih.gov/geo/series/GSE57nnn/GSE57338/soft/GSE57338_family.soft.gz"
        )
        os.system(f"gunzip -f {SOFT}.gz")
    print("Parsing transcriptome …")
    gse = GEOparse.get_GEO(filepath=SOFT, annotate_gpl=False)
    expr = gse.pivot_samples("VALUE").T
    titles = gse.phenotype_data["title"]
    y = (~titles.str.contains("Non-failing", case=False)).astype(int).values
    print(f"  Transcriptome shape: {expr.shape}, labels: {np.bincount(y)}")
    return expr, y
# ─── 2) GENOMICS ───
def load_genomics(path="cad.add.160614.website.txt"):
    print("Loading GWAS summary stats …")
    df = pd.read_csv(path, sep=r"\s+", engine="python",
                     usecols=["markername","beta","p_dgc"])
    df["p_dgc"].replace(0,1e-300, inplace=True)
    df["logp"] = -np.log10(df["p_dgc"])
    top = df.nlargest(100, "logp").reset_index(drop=True)
    print(f"  Top SNPs: {top.markername.tolist()[:5]}…(+{len(top)-5})")
    return top
# ─── 3) METABOLOMICS ───
def load_metabolomics(path="ukb_nightingale_biomarker_atlas.csv"):
    print("Loading metabolomics atlas …")
    df = pd.read_csv(path)
    sel = df[(df.icd10=="I25") & (df.endpoint_type=="incident")].copy()
    sel["pvalue"] = sel["pvalue"].clip(lower=1e-300)
    sel["logp"]   = -np.log10(sel["pvalue"])
    topm = sel.nlargest(100, "logp").reset_index(drop=True)
    print(f"  Top metabolites: {topm.biomarker_id.tolist()[:5]}…(+{len(topm)-5})")
    return topm
def main():
    # 1. Load each omic view
    expr_df, y         = load_transcriptome()
    top_snp            = load_genomics()
    top_met            = load_metabolomics()
    # 2. Split transcriptome into train/test
    X_tr_e, X_te_e, y_tr, y_te = train_test_split(
        expr_df, y, test_size=0.2, stratify=y, random_state=0
    )
    # 3. Impute & scale transcriptome
    imp_e = SimpleImputer(strategy="median")
    sc_e  = StandardScaler()
    X_tr_e = sc_e.fit_transform(imp_e.fit_transform(X_tr_e))
    X_te_e = sc_e.transform(imp_e.transform(X_te_e))
    # 4. LASSO selection
    print("Running LASSO feature selection …")
    lasso = LogisticRegressionCV(
        penalty='l1', solver='saga',
        cv=5, scoring='roc_auc',
        max_iter=10000, tol=1e-4, n_jobs=-1
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        lasso.fit(X_tr_e, y_tr)
    sel = SelectFromModel(lasso, max_features=50, prefit=True)
    X_tr_sel = sel.transform(X_tr_e)
    X_te_sel = sel.transform(X_te_e)
    genes_sel = expr_df.columns[sel.get_support()]
    print(f"  Selected {len(genes_sel)} genes")
    # 5. Build genomic block
    G_tr = np.tile(top_snp.beta.values, (X_tr_sel.shape[0],1))
    G_te = np.tile(top_snp.beta.values, (X_te_sel.shape[0],1))
    sc_g = StandardScaler().fit(G_tr)
    G_tr, G_te = sc_g.transform(G_tr), sc_g.transform(G_te)
    snps_sel = top_snp.markername.tolist()
    # 6. Build metabolomic block
    M_tr = np.tile(top_met.logp.values, (X_tr_sel.shape[0],1))
    M_te = np.tile(top_met.logp.values, (X_te_sel.shape[0],1))
    sc_m = StandardScaler().fit(M_tr)
    M_tr, M_te = sc_m.transform(M_tr), sc_m.transform(M_te)
    mets_sel = top_met.biomarker_id.tolist()
    # 7. Fuse all
    X_tr = np.hstack([X_tr_sel, G_tr, M_tr])
    X_te = np.hstack([X_te_sel, G_te, M_te])
    print(f"Fused shapes → train: {X_tr.shape}, test: {X_te.shape}")
    # 8. Define models to compare
    models = {
        "Logistic L1": LogisticRegressionCV(
            penalty='l1', solver='saga',
            cv=5, scoring='roc_auc',
            max_iter=10000, tol=1e-4, n_jobs=-1
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, random_state=0, n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            use_label_encoder=False,
            eval_metric='logloss',
            n_estimators=300, verbosity=0
        ),
        "SVM": SVC(probability=True, kernel='rbf', random_state=0)
    }
    # 9. Train & evaluate each
    results = {}
    plt.figure(figsize=(6,5))
    for name, model in models.items():
        print(f"\n>>> {name}")
        # Train
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            model.fit(X_tr, y_tr)
        # Predict & prob
        y_pred = model.predict(X_te)
        y_prob = model.predict_proba(X_te)[:,1]
        # Metrics
        acc  = accuracy_score(y_te, y_pred)
        prec = precision_score(y_te, y_pred)
        rec  = recall_score(y_te, y_pred)
        f1   = f1_score(y_te, y_pred)
        roc_fpr, roc_tpr, _ = roc_curve(y_te, y_prob)
        roc_auc = auc(roc_fpr, roc_tpr)
        # Store
        results[name] = {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "roc_auc": roc_auc
        }
        # Print per‐model
        print(f" Accuracy : {acc:.3f}")
        print(f" Precision: {prec:.3f}")
        print(f" Recall   : {rec:.3f}")
        print(f" F1‐score : {f1:.3f}")
        print(f" ROC AUC  : {roc_auc:.3f}")
        # Plot ROC
        plt.plot(roc_fpr, roc_tpr, lw=2, label=f"{name} (AUC={roc_auc:.2f})")
    # Finalize ROC plot
    plt.plot([0,1],[0,1],"--",c="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Test ROC Curves Comparison")
    plt.legend(loc="lower right", fontsize="small")
    plt.tight_layout(); plt.savefig("roc_comparison.png"); plt.close()
    # Summary table
    df_res = pd.DataFrame(results).T
    print("\n=== Model comparison ===")
    print(df_res)
    # Save summary
    df_res.to_csv("model_comparison_metrics.csv", float_format="%.3f")
    print("\nSaved ROC plot and metrics table.")
if __name__ == "__main__":
    main()
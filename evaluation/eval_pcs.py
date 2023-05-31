# Code adapted from bootphon/measuring-regularities-in-word-embeddings
from util import *
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import auc
import argparse
import os


parser = argparse.ArgumentParser()
parser.add_argument("--embedding_dir", type=str)
parser.add_argument("--analogy_dir", type=str)
parser.add_argument("--save_results", type=str, default="results/pcs/")
parser.add_argument("--nb_perms", type=int, default=50)
args = parser.parse_args()

models = []; scores = {}

if os.path.isdir(args.embedding_dir):  
    for filename in os.listdir(args.embedding_dir):
        if ".txt" in filename:
            name = " ".join(filename.split(".")[:-1])
            print("# Loading " + name + "...")
            models.append((name, load_model(embedding_dir = os.path.join(args.embedding_dir, filename))))
else:
    models.append(load_model(args.embedding_dir))

names, pairs_sets = bats_names_pairs(dir=args.analogy_dir)
print("# Loaded models and analogy dataset")

roc = {}

for model in models:
    model_sim, model_neg = metrics_from_model(model[1], names, pairs_sets, nb_perms=args.nb_perms)
    model_roc_fpr, model_roc_tpr = compute_roc_curves(model_sim, model_neg, nb_perms=args.nb_perms)

    roc[model[0]] = (model_roc_fpr, model_roc_tpr)

print("# Computed ROC curves")

for index, name in enumerate(names):
    fpr_perms = [row[index] for row in column(list(roc.values()), 0)]
    tpr_perms = [row[index] for row in column(list(roc.values()), 1)]
    x = np.linspace(0, 1, len(min(fpr_perms, key=len)))

    fig = go.Figure()
    fig.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1)

    for model in roc:
        fpr = np.interp(x, np.linspace(0, 1, len(roc[model][0][index])), roc[model][0][index])
        tpr = np.interp(x, np.linspace(0, 1, len(roc[model][1][index])), roc[model][1][index])
        fig.add_trace(go.Scatter(x=fpr, y=tpr, name=model + f" (AUC = {auc(fpr, tpr):.4f})", mode='lines'))

    fig.update_layout(
        title=name,
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        yaxis=dict(scaleanchor="x", scaleratio=1),
        xaxis=dict(constrain='domain'),
        width=1200, height=1200,
        font={"size" : 16})
    fig.write_image(args.save_results + name + ".pdf")
print("# Saved results for analogies")

fpr_means = []; tpr_means = []

for model in roc:
    fpr, tpr = roc[model]
    x = np.linspace(0, 1, len(min(fpr, key=len)))
    
    fpr = [np.interp(x, np.linspace(0, 1, len(item)), item) for item in fpr]
    tpr = [np.interp(x, np.linspace(0, 1, len(item)), item) for item in tpr]

    fpr_means.append(np.mean(fpr, axis=0))
    tpr_means.append(np.mean(tpr, axis=0))

fig = go.Figure()
fig.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1)

for index, fpr_mean in enumerate(fpr_means):
    tpr_mean = tpr_means[index]
    fig.add_trace(go.Scatter(x=fpr_mean, y=tpr_mean, name=models[index][0], mode='lines'))
    print("AUC for {}: {}".format(models[index][0], np.around(auc(fpr_mean, tpr_mean), 4)))

fig.update_layout(
    title="",
    xaxis_title='False Positive Rate',
    yaxis_title='True Positive Rate',
    yaxis=dict(scaleanchor="x", scaleratio=1),
    xaxis=dict(constrain='domain'),
    width=1200, height=1200,
    font={"size" : 16})

fig.write_image(args.save_results + "total_roc.pdf")
print("# Saved results for total")
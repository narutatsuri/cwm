"""
Embedding files are expected to be TXT files that follow GloVe format.
"""
from util import *
import plotly.graph_objects as go
import argparse
import os


parser = argparse.ArgumentParser()
parser.add_argument("--embedding_dir", type=str)
parser.add_argument("--analogy_dir", type=str)
parser.add_argument("--save_results", type=str, default="results/das/")
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

for model in models:
    model_score = compute_das(model[1], names, pairs_sets)
    scores[model[0]] = model_score
    print("DAS for {}: {}".format(model[0], np.around(np.mean([item for sublist in model_score for item in sublist]), 4)))

for name in names:
    fig = go.Figure()
    for score in scores:
        fig.add_trace(go.Box(y=scores[score][name], name=score))

    fig.update_layout(title=name, 
                      xaxis_title='', yaxis_title='Deviation',
                      yaxis_range=[0,1],
                      width=1200, height=1200)
    fig.write_image(args.save_results + name + ".pdf")

fig = go.Figure()
for model_name in scores:
    deviations = [model_name, [item for sublist in scores[model_name].values() for item in sublist]]
    for deviation in deviations:
        fig.add_trace(go.Box(y=deviation[1], name=deviation[0]))

fig.update_layout(title="Total Deviations",
                  xaxis_title='', yaxis_title='Deviation',
                  width=1200, height=1200)

fig.write_image(args.save_results + "total_deviation.pdf")
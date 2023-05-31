import gensim
from gensim import utils
import json
import os
import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import roc_curve
import torch


def column(matrix, i):
    return [row[i] for row in matrix]

def load_model(embedding_dir):
    model = gensim.models.KeyedVectors.load_word2vec_format(embedding_dir)

    return model

def bats_names_pairs(dir):
    names = []
    pairs_sets = []

    for d in os.listdir(dir):
        if os.path.isdir(os.path.join(dir,str(d))):
            for f in os.listdir(os.path.join(dir,str(d))):
                names.append(str(f)[:-4])
                pairs_sets.append(set())
                with utils.open_file(os.path.join(dir,str(d),str(f))) as fin:
                    for _, line in enumerate(fin):
                        line = utils.to_unicode(line)
                        a, b = [word.lower() for word in line.split()]
                        list_b = b.split("/")
                        if list_b[0] != a:
                            pairs_sets[-1].add((a, list_b[0]))

    return names, pairs_sets

def compute_alignment(vectors):
    # Normalize all vectors and compute mean
    vectors = [i/np.linalg.norm(i) for i in vectors]
    vec_mean = np.mean(vectors, axis=0)

    deviations = [np.dot(i, vec_mean) for i in vectors]

    return tuple(deviations)

def compute_das(model, names, pairs_sets):
    vocab_set = set(list(model.index_to_key))
    pairs_sets = [[d for d in list(pairs_sets[i]) if d[0] in vocab_set and d[1] in vocab_set] for i in range(len(pairs_sets))]
    name_to_score = {}
    for index, pair_set in tqdm(enumerate(pairs_sets), leave=False):
        vectors = []
        for word_pair in pair_set:
            vectors.append(model[word_pair[1]] - model[word_pair[0]])

        name_to_score[names[index]] = compute_alignment(vectors)
    return name_to_score

def metrics_from_model(model, names, pairs_sets, nb_perms):
    """
    """
    vocab_set = set(list(model.index_to_key))
    pairs_sets = [[d for d in list(pairs_sets[i]) if d[0] in vocab_set and d[1] in vocab_set] for i in range(len(pairs_sets))]

    normal_offsets = [[offset(model, i[0], i[1], names[k]) for i in pairs_sets[k]] for k in range(len(pairs_sets))]

    shf_offsets = []
    for k in range(len(pairs_sets)):
        perm_offsets = []
        for _ in range(nb_perms):
            perm_list = permutation_onecycle_avoidtrue(len(pairs_sets[k]), pairs_sets[k])
            offs = [offset(model, pairs_sets[k][i][0], pairs_sets[k][perm_list[i]][1], names[k]) for i in range(len(pairs_sets[k]))]
            perm_offsets.append(offs)
        shf_offsets.append(perm_offsets)

    similarities = similarity_offsets(normal_offsets)
    similarities_shuffle = []
    for perm in tqdm(range(nb_perms), leave=False):
        similarities_shuffle.append(similarity_offsets(column(shf_offsets, perm)))

    return similarities, similarities_shuffle

def permutation_onecycle(n):
    n1, n2 = 0, n
    l = np.random.permutation(range(n1, n2))
    for i in range(n1, n2):
        if i==l[i-n1]:
            j=np.random.randint(n1, n2)
            while j==l[j-n1]:
                j=np.random.randint(n1, n2)
            l[i-n1], l[j-n1] = l[j-n1], l[i-n1]
    return l

def permutation_onecycle_avoidtrue(n, real):
    test = False
    perm = permutation_onecycle(n)
    for i_r in range(len(real)):
        if real[i_r][1] == real[perm[i_r]][1]:
            test = True
    while test:
        test = False
        perm = permutation_onecycle(n)
        for i_r in range(len(real)):
            if real[i_r][1] == real[perm[i_r]][1]:
                test = True
    return perm

def similarity_offsets(list_offsets):
    sim_offsets = []
    for i in range(len(list_offsets)):
        sim_offsets.append([])
        list_tuples = list(list_offsets[i])
        for j in range(len(list_tuples)):
            for k in range(j+1,len(list_tuples)):
                sim_offsets[-1].append(cosine_similarity([list_tuples[j]], [list_tuples[k]])[0][0])
    return sim_offsets

def offset(model, w1, w2, name):
    def sublist(liste, pattern):
        indx = -1
        for i in range(len(liste)):
            if liste[i] == pattern[0] and liste[i:i+len(pattern)] == pattern:
                indx = i
        return indx
    
    def word_embedding(model, word):
        if type(model) == list:
            # BERT or GPT-2
            model, tokenizer = model
            tokenized_text = tokenizer.tokenize(word)
            indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
            embeds = np.array([model[i] for i in indexed_tokens])
            embedding = np.mean(embeds, axis=0)
        else:
            # gensim based model
            embedding = model.get_vector(word)
        return embedding
    
    if type(model) == list and len(model) == 3:
        model, tokenizer, model_name = model

        with open(os.path.join("BATS_3.0", "context_sentences.json")) as json_file:
            context_sentences = json.load(json_file)
        context = context_sentences[name[:3]]

        c1, c2 = context

        sentence = " ".join([c1, w1, c2, w2])
        if model_name == "gpt-context":
            sentence = "[CLS] " + sentence + " [SEP]"
        else:
            w1 = " "+w1
            w2 = " "+w2

        tokenized_sentence = tokenizer.tokenize(sentence)
        tokenized_w1 = tokenizer.tokenize(w1)
        tokenized_w2 = tokenizer.tokenize(w2)

        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_sentence)
        tokens_tensor = torch.tensor([indexed_tokens])

        with torch.no_grad():
            if model_name == "gpt-context":
                segments_ids = [1] * len(tokenized_sentence)
                segments_tensors = torch.tensor([segments_ids])
                outputs = model(tokens_tensor, segments_tensors)
            else:
                outputs = model(tokens_tensor)
            hidden_states = outputs[2]

        token_embeddings = torch.stack(hidden_states, dim=0)
        token_embeddings = torch.squeeze(token_embeddings, dim=1)
        token_embeddings = token_embeddings.permute(1, 0, 2)

        token_vecs = []
        for token in token_embeddings:
            cat_vec = torch.cat((token[-1], token[-2], token[-3], token[-4]), dim=0)
            token_vecs.append(cat_vec)

        idx_w1 = sublist(tokenized_sentence, tokenized_w1)
        idx_w2 = sublist(tokenized_sentence, tokenized_w2)
        len_w1 = len(tokenized_w1)
        len_w2 = len(tokenized_w2)

        embd_w1 = torch.mean(torch.stack(token_vecs[idx_w1:idx_w1 + len_w1 + 1]), dim=0)
        embd_w2 = torch.mean(torch.stack(token_vecs[idx_w2:idx_w2 + len_w2 + 1]), dim=0)

        return embd_w2 - embd_w1

    else:
        return word_embedding(model, w2) - word_embedding(model, w1)
    
def compute_roc_curves(similarities, similarities_shuffle, nb_perms):
    roc_fpr = []; roc_tpr = []
    for i in range(len(similarities)):
        fpr_perm = []; tpr_perm = []
        for perm in range(nb_perms):
            y_true = [1 for j in range(len(similarities[i]))]+[0 for j in range(len(similarities_shuffle[perm][i]))]
            y_scores = list(similarities[i]) + list(similarities_shuffle[perm][i])

            fpr, tpr, _ = roc_curve(y_true,y_scores)
            fpr_perm.append(fpr); tpr_perm.append(tpr)

        x = np.linspace(0, 1, len(min(fpr_perm, key=len)))

        roc_fpr.append(np.mean([np.interp(x, np.linspace(0, 1, len(i)), i) for i in fpr_perm], axis=0))
        roc_tpr.append(np.mean([np.interp(x, np.linspace(0, 1, len(i)), i) for i in tpr_perm], axis=0))

    return roc_fpr, roc_tpr
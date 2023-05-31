// Code adapted from github.com/tmikolov/word2vec and github.com/yumeng5/Spherical-Text-Embedding

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include <time.h>
#include <unistd.h>
#include <stdbool.h>

#define MAX_STRING 100
#define MAX_SENTENCE_LENGTH 1000

const int vocab_hash_size = 30000000;
const int corpus_max_size = 40000000;

struct vocab_word {
    long long cn;
    char *word;
};

char train_file[MAX_STRING], load_emb_file[MAX_STRING];
char word_emb[MAX_STRING], context_emb[MAX_STRING], doc_output[MAX_STRING];
char save_vocab_file[MAX_STRING], read_vocab_file[MAX_STRING];
struct vocab_word *vocab;
int window = 5, min_count = 5, num_threads = 20, min_reduce = 1;
int *vocab_hash;
long long *doc_sizes;
long long vocab_max_size = 1000, vocab_size = 0, corpus_size = 0, layer1_size = 100;
long long train_words = 0, word_count_actual = 0, iter = 10, file_size = 0;
int negative = 2;
const int table_size = 1e8;
int *word_table;
float alpha = 0.04, starting_alpha, sample = 1e-3, margin = 0.15, lambda_1, lambda_2;
float *syn0;
bool save_after_iters = false;
clock_t start;


void InitUnigramTable() {
  int a, i;
  double train_words_pow = 0;
  double d1, power = 0.75;
  word_table = (int *) malloc(table_size * sizeof(int));
  for (a = 0; a < vocab_size; a++) train_words_pow += pow(vocab[a].cn, power);
  i = 0;
  d1 = pow(vocab[i].cn, power) / train_words_pow;
  for (a = 0; a < table_size; a++) {
    word_table[a] = i;
    if (a / (double) table_size > d1) {
      i++;
      d1 += pow(vocab[i].cn, power) / train_words_pow;
    }
    if (i >= vocab_size) i = vocab_size - 1;
  }
}

void ReadWord(char *word, FILE *fin) {
  int a = 0, ch;
  while (!feof(fin)) {
    ch = fgetc(fin);
    if (ch == 13) continue;
    if ((ch == ' ') || (ch == '\t') || (ch == '\n')) {
      if (a > 0) {
        if (ch == '\n') ungetc(ch, fin);
        break;
      }
      if (ch == '\n') {
        strcpy(word, (char *) "</s>");
        return;
      } else continue;
    }
    word[a] = ch;
    a++;
    if (a >= MAX_STRING - 1) a--;
  }
  word[a] = 0;
}

// Returns hash value of a word
int GetWordHash(char *word) {
  unsigned long long a, hash = 0;
  for (a = 0; a < strlen(word); a++) hash = hash * 257 + word[a];
  hash = hash % vocab_hash_size;
  return hash;
}

// Returns position of a word in the vocabulary; if the word is not found, returns -1
int SearchVocab(char *word) {
  unsigned int hash = GetWordHash(word);
  while (1) {
    if (vocab_hash[hash] == -1) return -1;
    if (!strcmp(word, vocab[vocab_hash[hash]].word)) return vocab_hash[hash];
    hash = (hash + 1) % vocab_hash_size;
  }
  return -1;
}

// Locate line number of current file pointer
int FindLine(FILE *fin) {
  long long pos = ftell(fin);
  long long lo = 0, hi = corpus_size - 1;
  while (lo < hi) {
    long long mid = lo + (hi - lo) / 2;
    if (doc_sizes[mid] > pos) {
      hi = mid;
    } else {
      lo = mid + 1;
    }
  }
  return lo;
}

// Reads a word and returns its index in the vocabulary
int ReadWordIndex(FILE *fin) {
  char word[MAX_STRING];
  ReadWord(word, fin);
  if (feof(fin)) return -1;
  return SearchVocab(word);
}

// Adds a word to the vocabulary
int AddWordToVocab(char *word) {
  unsigned int hash, length = strlen(word) + 1;
  if (length > MAX_STRING) length = MAX_STRING;
  vocab[vocab_size].word = (char *) calloc(length, sizeof(char));
  strcpy(vocab[vocab_size].word, word);
  vocab[vocab_size].cn = 0;
  vocab_size++;
  if (vocab_size + 2 >= vocab_max_size) {
    vocab_max_size += 1000;
    vocab = (struct vocab_word *) realloc(vocab, vocab_max_size * sizeof(struct vocab_word));
  }
  hash = GetWordHash(word);
  while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
  vocab_hash[hash] = vocab_size - 1;
  return vocab_size - 1;
}

int VocabCompare(const void *a, const void *b) {
  if (((struct vocab_word *) b)->cn == ((struct vocab_word *) a)->cn) return strcmp(((struct vocab_word *) b)->word, ((struct vocab_word *) a)->word);
  return ((struct vocab_word *) b)->cn - ((struct vocab_word *) a)->cn;
}

int IntCompare(const void * a, const void * b){
  return ( *(int*)a - *(int*)b );
}

// Sorts the vocabulary by frequency using word counts
void SortVocab() {
  int a, size;
  unsigned int hash;
  qsort(&vocab[1], vocab_size - 1, sizeof(struct vocab_word), VocabCompare);
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  size = vocab_size;
  train_words = 0;
  for (a = 0; a < size; a++) {
    if ((vocab[a].cn < min_count) && (a != 0)) {
      vocab_size--;
      free(vocab[a].word);
    } else {
      hash = GetWordHash(vocab[a].word);
      while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
      vocab_hash[hash] = a;
      train_words += vocab[a].cn;
    }
  }
  vocab = (struct vocab_word *) realloc(vocab, (vocab_size + 1) * sizeof(struct vocab_word));
}

// Reduces the vocabulary by removing infrequent tokens
void ReduceVocab() {
  int a, b = 0;
  unsigned int hash;
  for (a = 0; a < vocab_size; a++)
    if (vocab[a].cn > min_reduce) {
      vocab[b].cn = vocab[a].cn;
      vocab[b].word = vocab[a].word;
      b++;
    } else free(vocab[a].word);
  vocab_size = b;
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  for (a = 0; a < vocab_size; a++) {
    hash = GetWordHash(vocab[a].word);
    while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
    vocab_hash[hash] = a;
  }
  fflush(stdout);
  min_reduce++;
}

void LearnVocabFromTrainFile() {
  char word[MAX_STRING];
  FILE *fin;
  long long a, i;
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  fin = fopen(train_file, "rb");
  if (fin == NULL) {
    printf("[ERROR] Training file not found. \n");
    exit(1);
  }
  vocab_size = 0;
  AddWordToVocab((char *) "</s>");

  while (1) {
    ReadWord(word, fin);
    if (feof(fin)) break;
    train_words++;
    if (train_words % 100000 == 0) {
      printf("%lldK%c", train_words / 1000, 13);
      fflush(stdout);
    }
    i = SearchVocab(word);
    if (i == -1) {
      a = AddWordToVocab(word);
      vocab[a].cn = 1;
    }
    else if (i == 0) {
      vocab[i].cn++;
      doc_sizes[corpus_size] = ftell(fin);
      corpus_size++;
      if (corpus_size >= corpus_max_size) {
        printf("[ERROR] No. of documents in corpus is larger than \"corpus_max_size\". \n");
        exit(1);
      }
    }
    else vocab[i].cn++;
    if (vocab_size > vocab_hash_size * 0.7) ReduceVocab();
  }
  SortVocab();

  file_size = ftell(fin);
  fclose(fin);
}

void SaveVocab() {
  long long i;
  FILE *fo = fopen(save_vocab_file, "wb");
  for (i = 0; i < vocab_size; i++) fprintf(fo, "%s %lld\n", vocab[i].word, vocab[i].cn);
  fclose(fo);
}

void ReadVocab() {
  long long a, i = 0;
  char c;
  char word[MAX_STRING];
  FILE *fin = fopen(read_vocab_file, "rb");
  if (fin == NULL) {
    printf("[ERROR] Vocabulary file not found. \n");
    exit(1);
  }
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  vocab_size = 0;
  while (1) {
    ReadWord(word, fin);
    if (feof(fin)) break;
    a = AddWordToVocab(word);
    fscanf(fin, "%lld%c", &vocab[a].cn, &c);
    i++;
  }
  SortVocab();

  fin = fopen(train_file, "rb");
  if (fin == NULL) {
    printf("[ERROR] Training data file not found. \n");
    exit(1);
  }
  fseek(fin, 0, SEEK_END);
  file_size = ftell(fin);
  fclose(fin);
}

void LoadEmb(char *emb_file, float *emb_ptr) {
  long long a, b;
  int *vocab_match_tmp = (int *) calloc(vocab_size, sizeof(int));
  int vocab_size_tmp = 0, word_dim;
  char *current_word = (char *) calloc(MAX_STRING, sizeof(char));
  float *syn_tmp = NULL, norm;
  unsigned long long next_random = 1;
  a = posix_memalign((void **) &syn_tmp, 128, (long long) layer1_size * sizeof(float));
  if (syn_tmp == NULL) {
    printf("[ERROR] Memory allocation failed. \n");
    exit(1);
  }
  printf("Loading embedding from file... %s\n", emb_file);
  if (access(emb_file, R_OK) == -1) {
    printf("[ERROR] File %s does not exist. \n", emb_file);
    exit(1);
  }
  // read embedding file
  FILE *fp = fopen(emb_file, "r");
  fscanf(fp, "%d", &vocab_size_tmp);
  fscanf(fp, "%d", &word_dim);
  if (layer1_size != word_dim) {
    printf("[ERROR] Embedding dimension incompatible with pretrained file. \n");
    exit(1);
  }
  vocab_size_tmp = 0;
  while (1) {
    fscanf(fp, "%s", current_word);
    a = SearchVocab(current_word);
    if (a == -1) {
      for (b = 0; b < layer1_size; b++) fscanf(fp, "%f", &syn_tmp[b]);
    }
    else {
      for (b = 0; b < layer1_size; b++) fscanf(fp, "%f", &emb_ptr[a * layer1_size + b]);
      vocab_match_tmp[vocab_size_tmp] = a;
      vocab_size_tmp++;
    }
    if (feof(fp)) break;
  }
  printf("In vocab: %d\n", vocab_size_tmp);
  qsort(&vocab_match_tmp[0], vocab_size_tmp, sizeof(int), IntCompare);
  vocab_match_tmp[vocab_size_tmp] = vocab_size;
  int i = 0;
  for (a = 0; a < vocab_size; a++) {
    if (a < vocab_match_tmp[i]) {
      norm = 0.0;
      for (b = 0; b < layer1_size; b++) {
        next_random = next_random * (unsigned long long) 25214903917 + 11;
        emb_ptr[a * layer1_size + b] = (((next_random & 0xFFFF) / (float) 65536) - 0.5) / layer1_size;
        norm += emb_ptr[a * layer1_size + b] * emb_ptr[a * layer1_size + b];
      }
      for (b = 0; b < layer1_size; b++)
        emb_ptr[a * layer1_size + b] /= sqrt(norm);
    }
    else if (i < vocab_size_tmp) {
      i++;
    }
  }
  fclose(fp);
  free(current_word);
  free(emb_file);
  free(vocab_match_tmp);
  free(syn_tmp);
}

void InitNet() {
  long long a, b;
  unsigned long long next_random = 1;
  a = posix_memalign((void **) &syn0, 128, (long long) vocab_size * layer1_size * sizeof(float));
  if (syn0 == NULL) {
    printf("[ERROR] Memory allocation failed. \n");
    exit(1);
  }

  float norm;
  if (load_emb_file[0] != 0) {
    char *center_emb_file = (char *) calloc(MAX_STRING, sizeof(char));
    strcpy(center_emb_file, load_emb_file);
    strcat(center_emb_file, "_w.txt");
    LoadEmb(center_emb_file, syn0);
  }
  else {
    for (a = 0; a < vocab_size; a++) {
      norm = 0.0;
      for (b = 0; b < layer1_size; b++) {
        next_random = next_random * (unsigned long long) 25214903917 + 11;
        syn0[a * layer1_size + b] = (((next_random & 0xFFFF) / (float) 65536) - 0.5) / layer1_size;
        norm += syn0[a * layer1_size + b] * syn0[a * layer1_size + b];
      }
      for (b = 0; b < layer1_size; b++)
        syn0[a * layer1_size + b] /= sqrt(norm);
    }
  }
}

void *TrainModelThread(void *id) {
  long long a, b, d, doc = 0, word, last_word, sentence_length = 0, sentence_position = 0;
  long long word_count = 0, last_word_count = 0, sen[MAX_SENTENCE_LENGTH + 1];
  long long l1, l2, l3 = 0, c, target;
  unsigned long long next_random = (long long) id;
  float u_norm, v_norm, v_prime_norm, uv_cos, uv_prime_cos;
  float obj_w = 0;
  clock_t now;
  float *u_grad = (float *) calloc(layer1_size, sizeof(float));
  float *v_grad = (float *) calloc(layer1_size, sizeof(float));
  float *v_prime_grad = (float *) calloc(layer1_size, sizeof(float));
  float progress, estimated_time;
  int sec, h, m, s;
  FILE *fi = fopen(train_file, "rb");
  fseek(fi, file_size / (long long) num_threads * (long long) id, SEEK_SET);

  while (1) {
    if (word_count - last_word_count > 10000) {
      word_count_actual += word_count - last_word_count;
      last_word_count = word_count;
      now = clock();
      progress = word_count_actual / (float) (iter * train_words + 1) * 100;

      printf("%cAlpha: %-8f  Loss: %-8f  Progress: %-7.2f%%",
              13, 
              alpha,
              obj_w, 
              progress);
      fflush(stdout);

      alpha = starting_alpha * (1 - word_count_actual / (float) (iter * train_words + 1));
      if (alpha < starting_alpha * 0.0001) alpha = starting_alpha * 0.0001;
    }
    if (sentence_length == 0) {
      doc = FindLine(fi);
      while (1) {
        word = ReadWordIndex(fi);
        if (feof(fi)) break;
        if (word == -1) continue;
        word_count++;
        if (word == 0) break;
        if (sample > 0) {
          float ran = (sqrt(vocab[word].cn / (sample * train_words)) + 1) * (sample * train_words) /
                     vocab[word].cn;
          next_random = next_random * (unsigned long long) 25214903917 + 11;
          if (ran < (next_random & 0xFFFF) / (float) 65536) continue;
        }
        sen[sentence_length] = word;
        sentence_length++;
        if (sentence_length >= MAX_SENTENCE_LENGTH) break;
      }
      sentence_position = 0;
    }

    if (feof(fi) || (word_count > train_words / num_threads)) {
      word_count_actual += word_count - last_word_count;
      break;
    }

    word = sen[sentence_position];
    if (word == -1) continue;
    for (c = 0; c < layer1_size; c++) u_grad[c] = 0;
    for (c = 0; c < layer1_size; c++) v_grad[c] = 0;
    for (c = 0; c < layer1_size; c++) v_prime_grad[c] = 0;
    next_random = next_random * (unsigned long long) 25214903917 + 11;
    b = next_random % window;

    for (a = b; a < window * 2 + 1 - b; a++)
      if (a != window) {
        c = sentence_position - window + a;
        if (c < 0) continue;
        if (c >= sentence_length) continue;
        last_word = sen[c];
        if (last_word == -1) continue;
        // Center word u
        l1 = last_word * layer1_size;

        obj_w = 0;
        for (d = 0; d < negative + 1; d++) {
          if (d == 0) {
            // Positive window word v
            l2 = word * layer1_size;
          } else {
            next_random = next_random * (unsigned long long) 25214903917 + 11;
            target = word_table[(next_random >> 16) % table_size];
            if (target == 0) target = next_random % (vocab_size - 1) + 1;
            if (target == word) continue;
            // Negative window word v'
            l3 = target * layer1_size;

            u_norm = 0, v_norm = 0, v_prime_norm = 0;
            uv_cos = 0, uv_prime_cos = 0;

            // Compute vector norms
            for (c = 0; c < layer1_size; c++) u_norm += syn0[c + l1] * syn0[c + l1]; u_norm = sqrt(u_norm);
            for (c = 0; c < layer1_size; c++) v_norm += syn0[c + l2] * syn0[c + l2]; v_norm = sqrt(v_norm);
            for (c = 0; c < layer1_size; c++) v_prime_norm += syn0[c + l3] * syn0[c + l3]; v_prime_norm = sqrt(v_prime_norm);

            // Compute cosine similarities
            for (c = 0; c < layer1_size; c++) uv_cos += (syn0[c + l1] / u_norm) * (syn0[c + l2] / v_norm);
            for (c = 0; c < layer1_size; c++) uv_prime_cos += (syn0[c + l1] / u_norm) * (syn0[c + l3] / v_prime_norm);

            if(margin - lambda_1 * uv_cos + lambda_2 * uv_prime_cos > 0){
              obj_w += margin - lambda_1 * uv_cos + lambda_2 * uv_prime_cos;

              // Compute gradients
              for (c = 0; c < layer1_size; c++) u_grad[c] = lambda_1 * (-1 * syn0[c + l2] / (u_norm * v_norm) + uv_cos * syn0[c + l1] / (u_norm * u_norm)) + 
                                                            lambda_2 * (syn0[c + l3] / (u_norm * v_prime_norm) - uv_prime_cos * syn0[c + l1] / (u_norm * u_norm));
              for (c = 0; c < layer1_size; c++) v_grad[c] = lambda_1 * (-1 * syn0[c + l1] / (u_norm * v_norm) + uv_cos * syn0[c + l2] / (v_norm * v_norm));
              for (c = 0; c < layer1_size; c++) v_prime_grad[c] = lambda_2 * (syn0[c + l1] / (u_norm * v_prime_norm) - uv_prime_cos * syn0[c + l3] / (v_prime_norm * v_prime_norm));

              // Update embeddings
              for (c = 0; c < layer1_size; c++) syn0[c + l1] -= alpha * u_grad[c];
              for (c = 0; c < layer1_size; c++) syn0[c + l2] -= alpha * v_grad[c];
              for (c = 0; c < layer1_size; c++) syn0[c + l3] -= alpha * v_prime_grad[c];
            }
          }
        }
      }

    sentence_position++;
    if (sentence_position >= sentence_length) {
      sentence_length = 0;
      continue;
    }
  }
  fclose(fi);
  free(u_grad);
  free(v_grad);
  free(v_prime_grad);
  pthread_exit(NULL);
}

void SaveEmbedding(int current_iter) {
  long a, b;
  FILE *fo;
  static char temp[4096];
  static char buffer[4096];

  char *p;
  char *orig = "iter-", *rep, iter_str[5];
  sprintf(iter_str, "%d", iter);
  char *result = malloc(strlen(orig) + strlen(iter_str) + 1);
  strcpy(result, orig);
  strcat(result, iter_str);

  char *orig = "iter-", *rep = "iter-";
  sprintf(iter_str, "%d", current_iter);
  char *result_2 = malloc(strlen(orig) + strlen(iter_str) + 1);
  strcpy(result_2, orig);
  strcat(result_2, iter_str);

  strcpy(temp, word_emb);
  strncpy(buffer, temp, p-temp);
  buffer[p-temp] = '\0';
  sprintf(buffer + (p - temp), "%s%s", result_2, p + strlen(result));
  sprintf(word_emb, "%s", buffer);    

  printf("\n# Writing trained embeddings to file... (iteration %d)\n", current_iter);
  fo = fopen(word_emb, "wb");
  fprintf(fo, "%lld %lld\n", vocab_size, layer1_size);
  for (a = 0; a < vocab_size; a++) {
    fprintf(fo, "%s ", vocab[a].word);
    for (b = 0; b < layer1_size; b++) {
      fprintf(fo, "%lf ", syn0[a * layer1_size + b]);
    }
    fprintf(fo, "\n");
  }
  fclose(fo);
}

void TrainModel() {
  long a, current_iter;
  pthread_t *pt = (pthread_t *) malloc(num_threads * sizeof(pthread_t));
  printf("Training using file: %s\n", train_file);

  starting_alpha = alpha;
  if (read_vocab_file[0] != 0) ReadVocab(); else LearnVocabFromTrainFile();
  if (save_vocab_file[0] != 0) SaveVocab();

  InitNet();
  InitUnigramTable();
  start = clock();

  printf("\n----------Training Parameters----------\n");
  printf("%-20s%-12lld\n", "Dimensions: ", layer1_size);
  printf("%-20s%-12lld\n", "Iterations: ", iter);
  printf("%-20s%-12lld\n", "Vocabulary Size: ", vocab_size);
  printf("%-20s%-12lld\n", "Total Word Count: ", train_words);  
  printf("%-20s%-12f\n", "Alpha: ", alpha);
  printf("%-20s%-12f\n", "Margin: ", margin);
  printf("%-20s%-12f\n", "Lambda_1: ", lambda_1);
  printf("%-20s%-12f\n", "Lambda_2: ", lambda_2);
  printf("%-20s%-12d\n", "Window Size: ", window);
  printf("%-20s%-12d\n", "Negative Samples: ", negative);
  printf("%-20s%-12f\n", "Subsampling Ratio: ", sample);
  printf("%-20s%-12d\n", "Minimum Count: ", min_count);
  printf("%-20s%-12d\n", "Threads: ", num_threads);
  printf("----------------------------------------\n");

  printf("Training... \n");
  for(current_iter = 1; current_iter <= iter; current_iter++){
    for (a = 0; a < num_threads; a++) pthread_create(&pt[a], NULL, TrainModelThread, (void *) a);
    for (a = 0; a < num_threads; a++) pthread_join(pt[a], NULL);
    if (save_after_iters) SaveEmbedding(current_iter);
  }
  SaveEmbedding(iter);
}

int ArgPos(char *str, int argc, char **argv) {
  int a;
  for (a = 1; a < argc; a++)
    if (!strcmp(str, argv[a])) {
      if (a == argc - 1) {
        printf("[ERROR] Argument missing for %s. \n", str);
        exit(1);
      }
      return a;
    }
  return -1;
}

int main(int argc, char **argv) {
  int i;
  word_emb[0] = 0;
  save_vocab_file[0] = 0;
  read_vocab_file[0] = 0;
  if ((i = ArgPos((char *) "-size", argc, argv)) > 0) layer1_size = atoi(argv[i + 1]);
  if ((i = ArgPos((char *) "-train", argc, argv)) > 0) strcpy(train_file, argv[i + 1]);
  if ((i = ArgPos((char *) "-save_vocab", argc, argv)) > 0) strcpy(save_vocab_file, argv[i + 1]);
  if ((i = ArgPos((char *) "-read_vocab", argc, argv)) > 0) strcpy(read_vocab_file, argv[i + 1]);
  if ((i = ArgPos((char *) "-load_emb", argc, argv)) > 0) strcpy(load_emb_file, argv[i + 1]);
  if ((i = ArgPos((char *) "-alpha", argc, argv)) > 0) alpha = atof(argv[i + 1]);
  if ((i = ArgPos((char *) "-word_output", argc, argv)) > 0) strcpy(word_emb, argv[i + 1]);
  if ((i = ArgPos((char *) "-window", argc, argv)) > 0) window = atoi(argv[i + 1]);
  if ((i = ArgPos((char *) "-sample", argc, argv)) > 0) sample = atof(argv[i + 1]);
  if ((i = ArgPos((char *) "-negative", argc, argv)) > 0) negative = atoi(argv[i + 1]);
  if ((i = ArgPos((char *) "-threads", argc, argv)) > 0) num_threads = atoi(argv[i + 1]);
  if ((i = ArgPos((char *) "-margin", argc, argv)) > 0) margin = atof(argv[i + 1]);
  if ((i = ArgPos((char *) "-lambda_1", argc, argv)) > 0) lambda_1 = atof(argv[i + 1]);
  if ((i = ArgPos((char *) "-lambda_2", argc, argv)) > 0) lambda_2 = atof(argv[i + 1]);
  if ((i = ArgPos((char *) "-iter", argc, argv)) > 0) iter = atoi(argv[i + 1]);
  if ((i = ArgPos((char *) "-save_after_iter", argc, argv)) > 0) save_after_iters = true;
  if ((i = ArgPos((char *) "-min_count", argc, argv)) > 0) min_count = atoi(argv[i + 1]);

  vocab = (struct vocab_word *) calloc(vocab_max_size, sizeof(struct vocab_word));
  vocab_hash = (int *) calloc(vocab_hash_size, sizeof(int));
  doc_sizes = (long long *) calloc(corpus_max_size, sizeof(long long));
  if (negative <= 0) {
    printf("[ERROR] Number of negative samples must be positive. \n");
    exit(1);
  }
  TrainModel();
  return 0;
}

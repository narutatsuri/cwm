cd ./model
gcc ${name}.c  -o ${name} -lm -pthread
cd ..

name=cwm
text_file=text.txt
margin=$1
lambda_1=$2
lambda_2=$3
alpha=$4
iter=$5
word_dim=$6
window_size=$7
negative=$8
threads=$9
min_count=50

out_file=${name}_${word_dim}d.txt

cd model
gcc ${name}.c  -o ${name} -lm -pthread
cd ..

home_dir=.

model/${name} \
-train ${home_dir}/${text_file} \
-word_output ${home_dir}/embeddings/${out_file} \
-size ${word_dim} \
-alpha ${alpha} \
-margin ${margin} \
-lambda_1 ${lambda_1} \
-lambda_2 ${lambda_2} \
-window ${window_size} \
-negative ${negative} \
-sample 1e-3 \
-min_count ${min_count} \
-iter ${iter} \
-threads ${threads}
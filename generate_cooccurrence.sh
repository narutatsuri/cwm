name=generate_cooccurrence
text_file=text.txt
min_count=50
window_size=5
out_file=cooccurrence_dataset_min_count-${min_count}_window_size-${window_size}.txt
threads=1

cd ./model
gcc ${name}.c  -o ${name} -lm -pthread
cd ..

start=$SECONDS

./src/${name} \
-train dataset/${text_file} \
-word-output cooccurrences/${out_file} \
-window ${window_size} \
-min-count ${min_count} \
-threads ${threads} \
#!/bin/bash

# 定义输出文件的路径
mkdir -p outputs/batch-deepfold
OUTPUT_FILE="outputs/batch-deepfold/batch-deepfold.log"
PROOF_SIZE_FILE="batch-deepfold-proofsize.csv"

echo "polyfrim benchmarking..."
# 运行cargo bench命令并将结果输出到文件
cargo bench -p batch -- --nocapture --quiet > $OUTPUT_FILE

# 检查命令是否成功执行
if [ $? -eq 0 ]; then
    echo "Benchmark results have been written to $OUTPUT_FILE"
else
    echo "Benchmark failed to run"
fi

echo "batch-deepfold proofsizing..."
cargo test -p batch --release -- --nocapture --quiet
cp batch/batch.csv outputs/batch-deepfold/$PROOF_SIZE_FILE
echo "batch-deepfold proofsize has been written to $PROOF_SIZE_FILE"




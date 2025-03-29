#!/bin/bash

# 定义输出文件的路径
mkdir -p outputs/basefold
OUTPUT_FILE="outputs/basefold/basefold.log"
PROOF_SIZE_FILE="basefold-proofsize.csv"

echo "Basefold benchmarking..."
# 运行cargo bench命令并将结果输出到文件
cargo bench -p basefold -- --nocapture --quiet > $OUTPUT_FILE

# 检查命令是否成功执行
if [ $? -eq 0 ]; then
    echo "Benchmark results have been written to $OUTPUT_FILE"
else
    echo "Benchmark failed to run"
fi

echo "basefold proofsizing..."
cargo test -p deepfold --release -- --nocapture --quiet
cp basefold/basefold.csv outputs/basefold/$PROOF_SIZE_FILE
echo "basefold proofsize has been written to $PROOF_SIZE_FILE"




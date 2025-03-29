#!/bin/bash

# 定义输出文件的路径
mkdir -p outputs/fri
OUTPUT_FILE="outputs/fri/fri.log"
PROOF_SIZE_FILE="fri-proofsize.csv"

echo "fri benchmarking..."
# 运行cargo bench命令并将结果输出到文件
cargo bench -p fri -- --nocapture --quiet > $OUTPUT_FILE

# 检查命令是否成功执行
if [ $? -eq 0 ]; then
    echo "Benchmark results have been written to $OUTPUT_FILE"
else
    echo "Benchmark failed to run"
fi

echo "fri proofsizing..."
cargo test -p fri --release -- --nocapture --quiet
cp fri/fri.csv outputs/fri/$PROOF_SIZE_FILE
echo "fri proofsize has been written to $PROOF_SIZE_FILE"




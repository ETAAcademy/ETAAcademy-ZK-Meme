#!/bin/bash

# 定义输出文件的路径
mkdir -p outputs/matmult
OUTPUT_FILE="outputs/matmult/matmult.log"
PROOF_SIZE_FILE="matmult-proofsize.csv"

echo "matmult benchmarking..."
# 运行cargo bench命令并将结果输出到文件
cargo bench -p matmult -- --nocapture --quiet > $OUTPUT_FILE

# 检查命令是否成功执行
if [ $? -eq 0 ]; then
    echo "Benchmark results have been written to $OUTPUT_FILE"
else
    echo "Benchmark failed to run"
fi

echo "matmult proofsizing..."
cargo test -p matmult --release -- --nocapture --quiet
cp -r matmult/* outputs/matmult/
echo "matmult proofsize has been written to $PROOF_SIZE_FILE"




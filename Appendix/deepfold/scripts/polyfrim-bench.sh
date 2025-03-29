#!/bin/bash

# 定义输出文件的路径
mkdir -p outputs/polyfrim
OUTPUT_FILE="outputs/polyfrim/polyfrim.log"
PROOF_SIZE_FILE="polyfrim-proofsize.csv"

echo "polyfrim benchmarking..."
# 运行cargo bench命令并将结果输出到文件
cargo bench -p polyfrim -- --nocapture --quiet > $OUTPUT_FILE

# 检查命令是否成功执行
if [ $? -eq 0 ]; then
    echo "Benchmark results have been written to $OUTPUT_FILE"
else
    echo "Benchmark failed to run"
fi

echo "polyfrim proofsizing..."
cargo test -p polyfrim --release -- --nocapture --quiet
cp polyfrim/polyfrim.csv outputs/polyfrim/$PROOF_SIZE_FILE
echo "polyfrim proofsize has been written to $PROOF_SIZE_FILE"




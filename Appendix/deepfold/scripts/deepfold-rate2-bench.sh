#!/bin/bash

# 更改CODE_RATE参数
mkdir -p outputs/deepfold-rate2
FILE_PATH="util/src/lib.rs"
sed -i 's/pub const CODE_RATE: usize = 3;/pub const CODE_RATE: usize = 1;/g' $FILE_PATH

# 定义输出文件的路径
OUTPUT_FILE="outputs/deepfold-rate2/deepfold-rate2.log"
PROOF_SIZE_FILE="deepfold-rate2-proofsize.csv"

echo "Deepfold benchmarking..."
# 运行cargo bench命令并将结果输出到文件
cargo bench -p deepfold -- --nocapture --quiet > $OUTPUT_FILE

# 检查命令是否成功执行
if [ $? -eq 0 ]; then
    echo "Benchmark results have been written to $OUTPUT_FILE"
else
    echo "Benchmark failed to run"
fi

echo "Deepfold proofsizing..."
cargo test -p deepfold --release -- --nocapture --quiet
cp deepfold/deepfold.csv outputs/deepfold-rate2/$PROOF_SIZE_FILE
echo "Deepfold proofsize has been written to $PROOF_SIZE_FILE"

sed -i 's/pub const CODE_RATE: usize = 1;/pub const CODE_RATE: usize = 3;/g' $FILE_PATH
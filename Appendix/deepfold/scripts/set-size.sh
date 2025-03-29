#!/bin/bash

# 检查是否提供了size参数
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <size>"
    exit 1
fi

# 读取size参数
size=$1

# 定义要替换的文件列表
FILE_PATH="util/src/lib.rs"
sed -i "s/pub const SIZE: usize = [0-9]*;/pub const SIZE: usize = $size;/g" $FILE_PATH

FILE_PATH="virgo/bench_gkr.py"
sed -i "s/SIZE = [0-9]*/SIZE = $size/g" $FILE_PATH
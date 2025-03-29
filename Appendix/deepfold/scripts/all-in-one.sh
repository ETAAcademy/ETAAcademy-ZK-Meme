rm -r ./outputs
mkdir outputs
./scripts/fri-bench.sh
./scripts/basefold-bench.sh
./scripts/polyfrim-bench.sh
./scripts/virgo-bench.sh
./scripts/deepfold-bench.sh
./scripts/deepfold-rate2-bench.sh
./scripts/batch-deepfold-bench.sh
./scripts/matmult-bench.sh
python scripts/bench-data-proc.py
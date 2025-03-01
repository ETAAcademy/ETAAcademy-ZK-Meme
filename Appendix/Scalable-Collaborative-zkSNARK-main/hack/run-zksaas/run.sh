# Check if IP address is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <server id> <log n> <l>"
    exit 1
fi

# Check if IP address is provided
if [ -z "$2" ]; then
    echo "Usage: $0 <server id> <log n> <l>"
    exit 1
fi

# Check if IP address is provided
if [ -z "$3" ]; then
    echo "Usage: $0 <server id> <log n> <l>"
    exit 1
fi

server_id=$1
n=$((2**$2))
l=$3
server_count=$((4*$l))

if ! which unzip &> /dev/null; then
    echo "unzip is not installed. Running the script..."
    sudo apt-get update
    sudo apt-get install unzip
else
    echo "unzip is installed. Skipping the script."
fi

unzip -o /tmp/tmp.zip -d /tmp/
rm /tmp/tmp.zip

cd /tmp/tmp
ps aux | awk '/plonk_bench/ && !/awk/ {print $2}' | xargs kill
sleep 1
/usr/bin/time -v ./plonk_bench $server_id ./network-address/$server_count $l $n | tee /tmp/zksaas.log
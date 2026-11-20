#!/usr/bin/env bash
# Fetch corpus sources: zhwiki text via HF parquet (hf-mirror), zhwikinews/zhwikivoyage via dumps.
export PATH=/root/miniconda3/bin:$PATH
cd /root/autodl-tmp/work || exit 1
mkdir -p data/raw data/raw/wiki_parquet logs

H=https://hf-mirror.com/datasets/wikimedia/wikipedia/resolve/main/20231101.zh
for i in 00000 00001; do
  f=train-$(printf '%05d' $((10#$i)))-of-00006.parquet
  python3 dl_parallel.py "$H/$f" "data/raw/wiki_parquet/$f" || echo "FAIL $f"
done

# news + voyage dumps (official dumps host, mirrored path)
B=https://dumps.wikimedia.your.org
for p in zhwikinews zhwikivoyage; do
  f=$p-latest-pages-articles.xml.bz2
  [ -s "data/raw/$f" ] || python3 dl_parallel.py "$B/$p/latest/$f" "data/raw/$f" || echo "FAIL $p"
done
echo SOURCES_DONE

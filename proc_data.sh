echo stripping review data
python3 read_amzn_gz_json.py ~/data/amazon/reviews_Automotive.json.gz > amzn_reviews_auto.txt
python3 read_amzn_gz_json.py ~/data/amazon/reviews_Beauty.json.gz > amzn_reviews_beauty.txt
python3 read_amzn_gz_json.py ~/data/amazon/reviews_Sports_and_Outdoors.json.gz > amzn_reviews_sport.txt

echo splitting test/train
python3 ~/code/cyberdevice/split_test_train.py 0.70 auto < amzn_reviews_auto.txt
python3 ~/code/cyberdevice/split_test_train.py 0.70 beauty < amzn_reviews_beauty.txt
python3 ~/code/cyberdevice/split_test_train.py 0.70 sport < amzn_reviews_sport.txt

echo catting data
cat *_train_*.csv > amzon_auto_beauty_sport_train.txt

echo building w2v
~/code/word2vec/bin/word2vec -train amzon_auto_beauty_sport_train.txt -size 50 -window 10 -negative 10 -sample 1e-5 -threads 4 -min-count 3 -output amzon_auto_beauty_sport_train.vectors -save-vocab amzon_auto_beauty_sport_train.vocab

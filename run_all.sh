python train_text_classifier.py --model cnn --dataset stsa.binary --word_vectors glove.840B.300d.txt --epoch 20
python run_dknn.py --model-setup results/stsa.binary_cnn/args.json
python train_text_classifier.py --model cnn --dataset mpqa --word_vectors glove.840B.300d.txt --epoch 20
python run_dknn.py --model-setup results/mpqa_cnn/args.json
python train_text_classifier.py --model cnn --dataset custrev --word_vectors glove.840B.300d.txt --epoch 20
python run_dknn.py --model-setup results/custrev_cnn/args.json
python train_text_classifier.py --model cnn --dataset subj --word_vectors glove.840B.300d.txt --epoch 20
python run_dknn.py --model-setup results/subj_cnn/args.json
python train_text_classifier.py --model cnn --dataset TREC --word_vectors glove.840B.300d.txt --epoch 20
python run_dknn.py --model-setup results/TREC_cnn/args.json


python train_text_classifier.py --model bilstm --dataset stsa.binary --word_vectors glove.840B.300d.txt --epoch 20
python run_dknn.py --model-setup results/stsa.binary_bilstm/args.json
python train_text_classifier.py --model bilstm --dataset mpqa --word_vectors glove.840B.300d.txt --epoch 20
python run_dknn.py --model-setup results/mpqa_bilstm/args.json
python train_text_classifier.py --model bilstm --dataset custrev --word_vectors glove.840B.300d.txt --epoch 20
python run_dknn.py --model-setup results/custrev_bilstm/args.json
python train_text_classifier.py --model bilstm --dataset subj --word_vectors glove.840B.300d.txt --epoch 20
python run_dknn.py --model-setup results/subj_bilstm/args.json
python train_text_classifier.py --model bilstm --dataset TREC --word_vectors glove.840B.300d.txt --epoch 20
python run_dknn.py --model-setup results/TREC_bilstm/args.json

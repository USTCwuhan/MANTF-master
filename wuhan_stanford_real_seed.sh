####################per 80################################
# neg_all
CUDA_VISIBLE_DEVICES=1 python train.py \
    --mode=train \
    --threads_num=6 \
    --n_epoch=1000 \
    --batch_size=128 \
    --random_seed=10 \
    --learning_rate=1e-4 \
    --optimizer=Adam \
    --model=MANTF \
    --company_num=35066 \
    --patent_num=257958 \
    --factor_size=500 \
    --dropout=0.5 \
    --patent_idx_path=data/stanford/samples/stanford_patent_index.txt \
    --company_idx_path=data/stanford/samples/stanford_company_index.txt \
    --comp_graph_path=data/stanford/features/stanford_percentage_wise_80_l_d_10_graph.npy \
    --comp_feat_path=data/stanford/features/stanford_comp_feat.npy \
    --comp_feat_freeze=n \
    --bert_size=768 \
    --gcn_dropout=0.2 \
    --pat_bert_title=data/stanford/features/stanford_pat_titles.npy \
    --pat_bert_abstract=data/stanford/features/stanford_pat_abstracts.npy \
    --pat_bert_claim=data/stanford/features/stanford_pat_avg_claims.npy \
    --meta_embed_path=data/stanford/features/stanford_patent_deepwalk_embedding.npy \
    --train_file=data/stanford/samples/stanford_percentage_wise_train_80 \
    --dev_file=data/stanford/samples/stanford_percentage_wise_val \
    --test_file=data/stanford/samples/stanford_percentage_wise_test \
    --save_every_epoch=1 \
    --begin_test_epoch=0 \
    --early_stop=10 \
    --neg_types 0 1 2 \
    --description='wuhan_stanford_real_loss012345_500_seed10'\
    --r_var=0.0001 \
    --p_var=0.01 \
    --u_var=0.01 \
    --v_var=0.01 \
    --g_var=0.01 \
    --q_var=0.01 \
    --lambda0=0.001 \

CUDA_VISIBLE_DEVICES=1 python train.py \
    --mode=train \
    --threads_num=6 \
    --n_epoch=1000 \
    --batch_size=128 \
    --random_seed=100 \
    --learning_rate=1e-4 \
    --optimizer=Adam \
    --model=MANTF \
    --company_num=35066 \
    --patent_num=257958 \
    --factor_size=500 \
    --dropout=0.5 \
    --patent_idx_path=data/stanford/samples/stanford_patent_index.txt \
    --company_idx_path=data/stanford/samples/stanford_company_index.txt \
    --comp_graph_path=data/stanford/features/stanford_percentage_wise_80_l_d_10_graph.npy \
    --comp_feat_path=data/stanford/features/stanford_comp_feat.npy \
    --comp_feat_freeze=n \
    --bert_size=768 \
    --gcn_dropout=0.2 \
    --pat_bert_title=data/stanford/features/stanford_pat_titles.npy \
    --pat_bert_abstract=data/stanford/features/stanford_pat_abstracts.npy \
    --pat_bert_claim=data/stanford/features/stanford_pat_avg_claims.npy \
    --meta_embed_path=data/stanford/features/stanford_patent_deepwalk_embedding.npy \
    --train_file=data/stanford/samples/stanford_percentage_wise_train_80 \
    --dev_file=data/stanford/samples/stanford_percentage_wise_val \
    --test_file=data/stanford/samples/stanford_percentage_wise_test \
    --save_every_epoch=1 \
    --begin_test_epoch=0 \
    --early_stop=10 \
    --neg_types 0 1 2 \
    --description='wuhan_stanford_real_loss012345_500_seed100'\
    --r_var=0.0001 \
    --p_var=0.01 \
    --u_var=0.01 \
    --v_var=0.01 \
    --g_var=0.01 \
    --q_var=0.01 \
    --lambda0=0.001 \

CUDA_VISIBLE_DEVICES=1 python train.py \
    --mode=train \
    --threads_num=6 \
    --n_epoch=1000 \
    --batch_size=128 \
    --random_seed=1000 \
    --learning_rate=1e-4 \
    --optimizer=Adam \
    --model=MANTF \
    --company_num=35066 \
    --patent_num=257958 \
    --factor_size=500 \
    --dropout=0.5 \
    --patent_idx_path=data/stanford/samples/stanford_patent_index.txt \
    --company_idx_path=data/stanford/samples/stanford_company_index.txt \
    --comp_graph_path=data/stanford/features/stanford_percentage_wise_80_l_d_10_graph.npy \
    --comp_feat_path=data/stanford/features/stanford_comp_feat.npy \
    --comp_feat_freeze=n \
    --bert_size=768 \
    --gcn_dropout=0.2 \
    --pat_bert_title=data/stanford/features/stanford_pat_titles.npy \
    --pat_bert_abstract=data/stanford/features/stanford_pat_abstracts.npy \
    --pat_bert_claim=data/stanford/features/stanford_pat_avg_claims.npy \
    --meta_embed_path=data/stanford/features/stanford_patent_deepwalk_embedding.npy \
    --train_file=data/stanford/samples/stanford_percentage_wise_train_80 \
    --dev_file=data/stanford/samples/stanford_percentage_wise_val \
    --test_file=data/stanford/samples/stanford_percentage_wise_test \
    --save_every_epoch=1 \
    --begin_test_epoch=0 \
    --early_stop=10 \
    --neg_types 0 1 2 \
    --description='wuhan_stanford_real_loss012345_500_seed1000'\
    --r_var=0.0001 \
    --p_var=0.01 \
    --u_var=0.01 \
    --v_var=0.01 \
    --g_var=0.01 \
    --q_var=0.01 \
    --lambda0=0.001 \
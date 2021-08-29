# coding: utf-8
import torch
USE_CUDA = torch.cuda.is_available()

batch_size = 64
embedding_size = 300
pos_embedding_size=128
hidden_size = 512
n_epochs = 140
learning_rate = 1e-3
weight_decay = 1e-5
beam_size = 3
n_layers = 2
Max_Question_len=120
Max_Expression_len=50
dataset="Math_23K"
#dataset="mawps""Math_23K""APE"

if dataset=="APE":
	n_epochs=50
else:
	n_epochs=140

#APE  word  彩电 是 两种 电视机
#APE  char 彩 电 是 两 种 电 视 机
USE_APE_word=True
USE_APE_char=False

#GTS USE_KAS2T_encoder=False
#KAS2T USE_KAS2T_encoder=True
USE_Glove_embedding=True
USE_KAS2T_encoder=True

#add our numeric encoder
USE_number_enc=False
USE_self_attn=False

if USE_number_enc==True:
	hidden_size=hidden_size+pos_embedding_size
if USE_self_attn==True:
	hidden_size=hidden_size+pos_embedding_size

#just symbol (baseline)
#USE_just_symbol=False
#just digit number(eg: 1 5 0)
USE_just_char_number=False

USE_Seq2Tree=True
USE_Seq2Seq=False

USE_gate=False
USE_cate=False
USE_compare=False


USE_G2T_stanford=True  #this is use gcn or not. has to be use unless baseline
USE_KA_graph=False
USE_G2T_graph=False

USE_dependency=True

USE_split=True
split_num=3
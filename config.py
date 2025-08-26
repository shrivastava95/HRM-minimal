from types import SimpleNamespace
from pathlib import Path

config = SimpleNamespace(
    causal             = False   ,
    norm_order         = 'post'  , # config.norm_order: 'pre' | 'post'
    norm_kind          = 'rms'   , # config.norm_kind : 'rms' | 'layer'
    activation         = 'swiglu', # config.activation: 'swiglu' | 'gelu' | 'silu'
    d_model            = 512     ,
    mlp_ratio          = 4.0     ,
    # ffn_hidden         = 2048    , # hidden dim for MLP -> d_model * mlp_ratio if not provided
    num_heads          = 8       ,
    num_puzzle_embs    = 1       , # one puzzle; one puzzle_emb
    vocab_size         = 11      ,
    L_depth            = 4       ,
    H_depth            = 4       ,
    L_cycles           = 2       ,
    H_cycles           = 2       ,
    M_min              = 2       ,
    M_max              = 16      ,
    max_seq_len        = 82      ,
    
    epochs             = 20000   ,
    eval_interval      = 2000    ,
    batch_size         = 32      ,
    lr_warmup_steps    = 2000    ,
    lr_min_ratio       = 1.0     ,
    lr                 = 2e-5    ,
    wd                 = 1e0     ,
    beta1              = 0.9     ,
    beta2              = 0.95    ,
    puzzle_emb_lr      = 2e-3    ,
    puzzle_emb_wd      = 1e0     ,
    rope_theta         = 10000.0 ,
    layer_norm_eps     = 1e-5    ,
    dropout            = 0.1     ,
    checkpoint_every   = True    ,
    ignore_index       = -100    ,
    
    train_data_path    = Path('./data') / 'sudoku-extreme-full' / 'train.csv',
    test_data_path     = Path('./data') / 'sudoku-extreme-full' / 'test.csv'  ,
)

assert config.norm_order in ['post', 'pre']

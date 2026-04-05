# 30 len
uv run python evaluate.py --checkpoint saved_models/inverse/optoformer_len_20_13m_10nm/best.pt --val_path data/max_len_30_10nm/val/part_000.arrow --beam_width 5 --n_samples 10000 --plot_dir eval_data/len_30_10nm/pi/epoch29/ --rope_scale_method pi --rope_scale_factor 1.5

uv run python evaluate.py --checkpoint saved_models/inverse/optoformer_len_20_13m_10nm/best.pt --val_path data/max_len_30_10nm/val/part_000.arrow --beam_width 5 --n_samples 10000 --plot_dir eval_data/len_30_10nm/ntk/epoch29/ --rope_scale_method ntk --rope_scale_factor 1.5

uv run python evaluate.py --checkpoint saved_models/inverse/optoformer_len_20_13m_10nm/best.pt --val_path data/max_len_30_10nm/val/part_000.arrow --beam_width 5 --n_samples 10000 --plot_dir eval_data/len_30_10nm/dynamic_ntk/epoch29/ --rope_scale_method dynamic_ntk --rope_scale_factor 1.5

uv run python evaluate.py --checkpoint saved_models/inverse/optoformer_len_20_13m_10nm/best.pt --val_path data/max_len_30_10nm/val/part_000.arrow --beam_width 5 --n_samples 10000 --plot_dir eval_data/len_30_10nm/yarn/epoch29/ --rope_scale_method yarn --rope_scale_factor 1.5

uv run python evaluate.py --checkpoint saved_models/inverse/optoformer_len_20_13m_10nm/best.pt --val_path data/max_len_30_10nm/val/part_000.arrow --beam_width 5 --n_samples 10000 --plot_dir eval_data/len_30_10nm/no_scale/epoch29/

# 40 len
uv run python evaluate.py --checkpoint saved_models/inverse/optoformer_len_20_13m_10nm/best.pt --val_path data/max_len_40_10nm/val/part_000.arrow --beam_width 5 --n_samples 10000 --plot_dir eval_data/len_40_10nm/pi/epoch29/ --rope_scale_method pi --rope_scale_factor 2.0

uv run python evaluate.py --checkpoint saved_models/inverse/optoformer_len_20_13m_10nm/best.pt --val_path data/max_len_40_10nm/val/part_000.arrow --beam_width 5 --n_samples 10000 --plot_dir eval_data/len_40_10nm/ntk/epoch29/ --rope_scale_method ntk --rope_scale_factor 2.0

uv run python evaluate.py --checkpoint saved_models/inverse/optoformer_len_20_13m_10nm/best.pt --val_path data/max_len_40_10nm/val/part_000.arrow --beam_width 5 --n_samples 10000 --plot_dir eval_data/len_40_10nm/dynamic_ntk/epoch29/ --rope_scale_method dynamic_ntk --rope_scale_factor 2.0

uv run python evaluate.py --checkpoint saved_models/inverse/optoformer_len_20_13m_10nm/best.pt --val_path data/max_len_40_10nm/val/part_000.arrow --beam_width 5 --n_samples 10000 --plot_dir eval_data/len_40_10nm/yarn/epoch29/ --rope_scale_method yarn --rope_scale_factor 2.0

uv run python evaluate.py --checkpoint saved_models/inverse/optoformer_len_20_13m_10nm/best.pt --val_path data/max_len_40_10nm/val/part_000.arrow --beam_width 5 --n_samples 10000 --plot_dir eval_data/len_40_10nm/no_scale/epoch29/

# 50 len
uv run python evaluate.py --checkpoint saved_models/inverse/optoformer_len_20_13m_10nm/best.pt --val_path data/max_len_50_10nm/val/part_000.arrow --beam_width 5 --n_samples 10000 --plot_dir eval_data/len_50_10nm/pi/epoch29/ --rope_scale_method pi --rope_scale_factor 2.5

uv run python evaluate.py --checkpoint saved_models/inverse/optoformer_len_20_13m_10nm/best.pt --val_path data/max_len_50_10nm/val/part_000.arrow --beam_width 5 --n_samples 10000 --plot_dir eval_data/len_50_10nm/ntk/epoch29/ --rope_scale_method ntk --rope_scale_factor 2.5

uv run python evaluate.py --checkpoint saved_models/inverse/optoformer_len_20_13m_10nm/best.pt --val_path data/max_len_50_10nm/val/part_000.arrow --beam_width 5 --n_samples 10000 --plot_dir eval_data/len_50_10nm/dynamic_ntk/epoch29/ --rope_scale_method dynamic_ntk --rope_scale_factor 2.5

uv run python evaluate.py --checkpoint saved_models/inverse/optoformer_len_20_13m_10nm/best.pt --val_path data/max_len_50_10nm/val/part_000.arrow --beam_width 5 --n_samples 10000 --plot_dir eval_data/len_50_10nm/yarn/epoch29/ --rope_scale_method yarn --rope_scale_factor 2.5

uv run python evaluate.py --checkpoint saved_models/inverse/optoformer_len_20_13m_10nm/best.pt --val_path data/max_len_50_10nm/val/part_000.arrow --beam_width 5 --n_samples 10000 --plot_dir eval_data/len_50_10nm/no_scale/epoch29/


# 10nm 20 seq len evals IN DISTRIBUTION
uv run python evaluate.py --checkpoint saved_models/inverse/optoformer_len_20_13m_10nm/best.pt --val_path data/max_len_20_10nm/val/part_000.arrow --beam_width 5 --n_samples 10000 --plot_dir eval_data/len_20_10nm/no_scale/epoch29/ 


# 5nm 20 seq len evals 
uv run python evaluate.py --checkpoint saved_models/inverse/optoformer_len_20_13m_10nm/best.pt --val_path data/max_len_20/val/part_000.arrow --beam_width 5 --n_samples 10000 --plot_dir eval_data/len_20_5nm/no_scale/epoch29/


# 15nm 20 len thick only evals
uv run python evaluate.py --checkpoint saved_models/inverse/optoformer_len_20_13m_10nm/best.pt --val_path data/thick/15nm/val/part_000.arrow --beam_width 5 --n_samples 10000 --plot_dir eval_data/len_20_15nm_thick/ntk/epoch29/ --rope_scale_method ntk --rope_scale_factor 1.5 --min_cum_depth 11000

uv run python evaluate.py --checkpoint saved_models/inverse/optoformer_len_20_13m_10nm/best.pt --val_path data/thick/15nm/val/part_000.arrow --beam_width 5 --n_samples 10000 --plot_dir eval_data/len_20_15nm_thick/dynamic_ntk/epoch29/ --rope_scale_method dynamic_ntk --rope_scale_factor 1.5 --min_cum_depth 11000

uv run python evaluate.py --checkpoint saved_models/inverse/optoformer_len_20_13m_10nm/best.pt --val_path data/thick/15nm/val/part_000.arrow --beam_width 5 --n_samples 10000 --plot_dir eval_data/len_20_15nm_thick/yarn/epoch29/ --rope_scale_method yarn --rope_scale_factor 1.5 --min_cum_depth 11000

uv run python evaluate.py --checkpoint saved_models/inverse/optoformer_len_20_13m_10nm/best.pt --val_path data/thick/15nm/val/part_000.arrow --beam_width 5 --n_samples 10000 --plot_dir eval_data/len_20_15nm_thick/no_scale/epoch29/ --min_cum_depth 11000

# 20nm 20 len thick only evals
uv run python evaluate.py --checkpoint saved_models/inverse/optoformer_len_20_13m_10nm/best.pt --val_path data/thick/20nm/val/part_000.arrow --beam_width 5 --n_samples 10000 --plot_dir eval_data/len_20_20nm_thick/ntk/epoch29/ --rope_scale_method ntk --rope_scale_factor 2.0 --min_cum_depth 11000

uv run python evaluate.py --checkpoint saved_models/inverse/optoformer_len_20_13m_10nm/best.pt --val_path data/thick/20nm/val/part_000.arrow --beam_width 5 --n_samples 10000 --plot_dir eval_data/len_20_20nm_thick/dynamic_ntk/epoch29/ --rope_scale_method dynamic_ntk --rope_scale_factor 2.0 --min_cum_depth 11000

uv run python evaluate.py --checkpoint saved_models/inverse/optoformer_len_20_13m_10nm/best.pt --val_path data/thick/20nm/val/part_000.arrow --beam_width 5 --n_samples 10000 --plot_dir eval_data/len_20_20nm_thick/yarn/epoch29/ --rope_scale_method yarn --rope_scale_factor 2.0 --min_cum_depth 11000

uv run python evaluate.py --checkpoint saved_models/inverse/optoformer_len_20_13m_10nm/best.pt --val_path data/thick/20nm/val/part_000.arrow --beam_width 5 --n_samples 10000 --plot_dir eval_data/len_20_20nm_thick/no_scale/epoch29/ --min_cum_depth 11000



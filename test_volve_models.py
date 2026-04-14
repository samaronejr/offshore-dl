import sys
sys.path.insert(0, 'src')
from scripts.run_optuna_hpo import _get_volve_models
models = _get_volve_models()
assert 'lstm' in models, "LSTM missing"
assert 'deeponet' in models, "DeepONet missing"
assert 'patchtst' in models, "PatchTST missing"
assert 'tcn' in models, "TCN missing"
for name, cfg in models.items():
    assert cfg['kwargs']['n_vars'] == 73, f"{name}: wrong n_vars"
    assert cfg['kwargs']['window_size'] == 90, f"{name}: wrong window_size"
assert 'target_channel' in models['patchtst']['kwargs'], "PatchTST missing target_channel"
assert models['patchtst']['kwargs']['target_channel'] == 48, "PatchTST wrong target_channel"
print('All checks passed')

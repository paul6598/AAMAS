"""Prepare, but do not launch, a strictly matched fixed-refresh RSVP control."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--reference',type=int,default=236)
    ap.add_argument('--out',type=Path,required=True)
    a = ap.parse_args()
    source = ROOT/'results/sacred'/str(a.reference)/'config.json'
    original = json.loads(source.read_text())
    assert original['env_args']['map_name']=='5m_vs_6m'
    assert original['scheduler']=='vf' and original['f_update']==200
    assert original['llm_cache'] is False and original['llm_temperature']==.2
    assert not original['use_action_masking'] and original['shaping_in_learner']
    assert original['lambda_floor_frac']==.4 and original['t_max']==1200000
    config = dict(original,runner='rsvp',name='rsvp',scheduler='fixed',
                  wandb_run='lehca-shape_F200_lam40_nc_t02_rsvp')
    # Keep even inactive scheduler knobs identical; only fixed/vf changes behavior.
    delta = {k:dict(before=original.get(k),after=v) for k,v in config.items()
             if original.get(k)!=v}
    assert set(delta)=={'runner','name','scheduler','wandb_run'}
    a.out.mkdir(parents=True,exist_ok=True)
    for name,obj in [('config.json',config),('manifest.json',dict(
        created_utc=datetime.now(timezone.utc).isoformat(),reference_sacred=a.reference,
        reference_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),delta=delta,
        budget=dict(t_env=1200000,wall_hours=24,seeds=[config['seed']]),
        code_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                     for directory in ['algorithm/rsvp','algorithm/lehca','env/semantic']
                     for p in (ROOT/directory).rglob('*.py')}))]:
        with (a.out/name).open('x') as f:json.dump(obj,f,indent=2)
    print(json.dumps(dict(config=str((a.out/'config.json').resolve()),delta=delta),indent=2))


if __name__=='__main__':main()

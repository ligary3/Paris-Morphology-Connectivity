from common import *
import configparser
import shutil
import hashlib

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
dest=HERE/'q_sensitivity'; dest.mkdir(exist_ok=True)
jobs=[]; audit=[]
# Same-q 112.5 replication first, then 16 changed-resistance solves.
for d,factor in [(112.5,1.)]+[(d,f) for f in (.5,2.) for d in DIRS]:
  for case in ('real','homo'):
    old=folder(d)/case/'circuitscape.ini'
    cfg=configparser.ConfigParser(); cfg.read(old,encoding='utf8')
    habitat=Path(cfg['Habitat raster or graph']['habitat_file'])
    focal=Path(cfg['Options for pairwise and one-to-all and all-to-one modes']['point_file'])
    a,h=cross.read_ascii(habitat); valid=a!=h['nodata_value']
    b=a.copy(); b[valid]=1+(a[valid]-1)/factor
    run=dest/f'q{factor:g}'/tag(d)/case; run.mkdir(parents=True,exist_ok=True)
    raster=run/'resistance.asc'; electrode=run/'focal.asc'
    with raster.open('w',encoding='ascii',newline='\n') as f:
      for k,v in h.items(): f.write(f'{k} {v:.12g}\n')
      np.savetxt(f,b,fmt='%.12g')
    shutil.copy2(focal,electrode)
    cfg['Habitat raster or graph']['habitat_file']=str(raster)
    cfg['Options for pairwise and one-to-all and all-to-one modes']['point_file']=str(electrode)
    cfg['Output options']['output_file']=str(run/'network.out')
    config=run/'circuitscape.ini'
    with config.open('w',encoding='utf8') as f: cfg.write(f)
    jobs.append(str(config))
    audit.append(dict(direction=d,q_factor=factor,case=case,n_valid=int(valid.sum()),
      min_resistance=float(b[valid].min()),max_resistance=float(b[valid].max()),
      mask_unchanged=bool(np.array_equal(a==h['nodata_value'],b==h['nodata_value'])),
      focal_sha256=sha(focal),copied_focal_sha256=sha(electrode),original_config=str(old)))
(dest/'jobs.txt').write_text('\n'.join(jobs),encoding='utf8')
save_json(dest/'input_audit.json',audit)
print('Prepared',len(jobs),'Circuit solves',flush=True)

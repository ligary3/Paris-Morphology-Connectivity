"""Retrieve public Copernicus GLO-30 tiles for the Paris locator basemap only."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import json
import requests
import numpy as np
import rasterio
from rasterio.merge import merge
from rasterio.warp import calculate_default_transform, reproject, Resampling

OUT=Path(__file__).resolve().parents[1]/'source_data/study_area/dem'
OUT.mkdir(parents=True,exist_ok=True)
def download(pair):
    lat,lon=pair
    name=f'Copernicus_DSM_COG_10_N{lat:02d}_00_E{lon:03d}_00_DEM'
    url=f'https://copernicus-dem-30m.s3.amazonaws.com/{name}/{name}.tif'
    dest=OUT/f'{name}.tif'
    if not dest.exists():
        print('Downloading',name,flush=True)
        with requests.get(url,stream=True,timeout=(20,120)) as r:
            r.raise_for_status()
            with dest.with_suffix('.part').open('wb') as f:
                for chunk in r.iter_content(1024*1024): f.write(chunk)
        dest.with_suffix('.part').replace(dest)
    print('Ready',name,flush=True)
    return dest,url

if __name__=='__main__':
    with ThreadPoolExecutor(max_workers=4) as pool:
        entries=list(pool.map(download,[(48,1),(48,2),(49,1),(49,2)]))
    srcs=[rasterio.open(p) for p,u in entries]
    bounds=(1.65,48.50,2.90,49.25)
    arr,tr=merge(srcs,bounds=bounds,nodata=-9999)
    transform,w,h=calculate_default_transform('EPSG:4326','EPSG:2154',arr.shape[2],arr.shape[1],*bounds,resolution=50)
    dst=np.full((h,w),-9999,dtype='float32')
    reproject(arr[0],dst,src_transform=tr,src_crs='EPSG:4326',dst_transform=transform,dst_crs='EPSG:2154',src_nodata=-9999,dst_nodata=-9999,resampling=Resampling.bilinear)
    for s in srcs:s.close()
    with rasterio.open(OUT/'paris_glo30_lamb93_50m.tif','w',driver='GTiff',width=w,height=h,count=1,dtype='float32',crs='EPSG:2154',transform=transform,nodata=-9999,compress='deflate') as f:f.write(dst,1)
    (OUT/'provenance.json').write_text(json.dumps({'product':'Copernicus GLO-30 public DSM, 2021 release on AWS','source_urls':[u for p,u in entries],'geographic_crop':bounds,'display_grid_m':50,'resampling':'bilinear','role':'cartographic context only; not WRF terrain input','source_documentation':'https://registry.opendata.aws/copernicus-dem/'},indent=2))
    valid=dst[dst!=-9999]
    print('Finished',dst.shape,float(valid.min()),float(valid.max()),flush=True)

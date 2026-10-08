"""Paris study-area map: geographic context only, no experimental results.

Run with the same Python environment as the Results figures.
All geographical layers and model boundaries are read from existing data.
"""

from release_paths import resolve_path as _release_path
from pathlib import Path
import sys
import json
import warnings
import shutil
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / '.runtime'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.colors import LightSource, LinearSegmentedColormap, Normalize
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd
import geopandas as gpd
import pyogrio
import shapely
from shapely.geometry import shape, box, Polygon
from pyproj import Transformer
import rasterio
from rasterio.features import shapes
import h5py

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'manuscript_figures'
DEST = ROOT / 'manuscript_paris/figures'
QA = OUT / 'qa/study_area'
CACHE = OUT / 'source_data/study_area'
for p in (DEST, QA, CACHE): p.mkdir(parents=True, exist_ok=True)
DATA = _release_path('E:/lunwen4/2')
MORPH = DATA / '建筑形态/Paris/processed/phase1a_v2'
RAW = next((DATA / '建筑形态/Paris/BDTOPO_3-0_TOUSTHEMES_GPKG_LAMB93_D075_2022-06-15').rglob('*.gpkg'))
CASE = Path('//wsl.localhost/Ubuntu-WRF/home/gary/wrf-work/cases/paris_geogrid_v1')
SKILL = _release_path('C:/Users/12169/.codex/skills/nature-figure/scripts')
sys.path.insert(0, str(SKILL))
from release_plot_checks import require_matplotlib_panel_alignment
warnings.filterwarnings('ignore', message='Field format')
plt.rcParams.update({'font.family':'Arial', 'font.size':8, 'axes.titlesize':9,
    'legend.fontsize':7, 'pdf.fonttype':42, 'svg.fonttype':'none',
    'axes.linewidth':.55, 'savefig.facecolor':'white'})
INK = '#344852'
WATER = '#8BC0D1'
GREEN = '#AAC4A3'
BUILD = '#8C9091'
CORE = '#AA5044'
CIRC = '#78699D'
CRS = 2154
PROJ = Transformer.from_crs(4326, CRS, always_xy=True)
O = np.array([652000., 6862000.])

def layer(name, columns=(), bbox=None):
    return pyogrio.read_dataframe(RAW, layer=name, columns=list(columns), bbox=bbox).to_crs(CRS)

def draw_geo(ax, data, **kw):
    if len(data):
        data.plot(ax=ax, aspect=None, rasterized=True, **kw)

def polygon_frame(geom):
    return gpd.GeoDataFrame(geometry=[geom], crs=CRS)

def fit(ax, bounds):
    x0,y0,x1,y1 = bounds
    p=ax.get_position(); r=p.width*ax.figure.get_figwidth()/(p.height*ax.figure.get_figheight())
    w=max(x1-x0,(y1-y0)*r); h=w/r
    cx=(x0+x1)/2; cy=(y0+y1)/2
    ax.set_xlim(cx-w/2,cx+w/2); ax.set_ylim(cy-h/2,cy+h/2)
    ax.set_aspect('equal',adjustable='datalim')
    ax.set_xticks([]);ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_color('#C4CCCE')

def panel(ax, letter, title):
    ax.text(0,1.025,f'({letter})',transform=ax.transAxes,weight='bold',fontsize=9,va='bottom',color='#000000')
    ax.text(.09,1.025,title,transform=ax.transAxes,fontsize=9,va='bottom',color='#000000')

def scale(ax, km, x=.07, y=.075):
    xl=ax.get_xlim(); yl=ax.get_ylim()
    xx=xl[0]+x*(xl[1]-xl[0]); yy=yl[0]+y*(yl[1]-yl[0])
    ax.plot([xx,xx+km*1000],[yy,yy],color=INK,lw=1.25,zorder=20)
    ax.annotate(f'{km} km',(xx+km*500,yy),xytext=(0,-4),textcoords='offset points',ha='center',va='top',fontsize=7,color=INK,zorder=21)

def north(ax):
    ax.annotate('',xy=(.947,.91),xytext=(.947,.81),xycoords='axes fraction',arrowprops={'arrowstyle':'-|>','lw':1,'color':INK},zorder=20)
    ax.text(.947,.94,'N',transform=ax.transAxes,ha='center',fontsize=8,color=INK)

def border(lon,lat):
    # Perimeter of the outer mass-grid centres; precise convention in caption.
    x=np.r_[lon[0],lon[1:,-1],lon[-1,-2::-1],lon[-2:0:-1,0],lon[0,0]]
    y=np.r_[lat[0],lat[1:,-1],lat[-1,-2::-1],lat[-2:0:-1,0],lat[0,0]]
    return np.array(PROJ.transform(x,y)).T

def domains():
    target=CACHE/'wrf_domains.npz'
    if not target.exists():
        out={}
        for n in (1,2,3):
            p=CASE/f'geo_em.d0{n}.nc'
            if n==3: p=MORPH/'wrf_ucp_d03_v1/geo_em.d03.nc'
            with h5py.File(p,'r',locking=False) as f:
                lat=f['XLAT_M'][0];lon=f['XLONG_M'][0]
                out[f'd0{n}']=border(lon,lat)
                if n==3:
                    out['x'],out['y']=PROJ.transform(lon,lat)
                    out['lu']=f['LU_INDEX'][0]
        np.savez_compressed(target,**out)
    return dict(np.load(target))

def circuit_domain():
    p=MORPH/'extended_domain_v1/circuit_real_homo_1125_v1/real/resistance_r1.asc'
    with rasterio.open(p) as src:
        valid=(src.read(1)!=src.nodata).astype('uint8')
        geom=shapely.union_all([shape(g) for g,v in shapes(valid,mask=valid.astype(bool),transform=src.transform) if v==1])
    return geom

def main():
    print('Reading geographic layers',flush=True)
    deps=layer('departement',['code_insee','nom_officiel'])
    paris=deps.loc[deps.code_insee.astype(str)=='75'].copy()
    assert len(paris)==1
    pgeom=paris.geometry.iloc[0]
    extent=box(*pgeom.bounds).buffer(1400).bounds
    read_extent=box(*extent).buffer(6000).bounds
    buildings=layer('batiment',bbox=read_extent)
    water=layer('surface_hydrographique',bbox=read_extent)
    veg=layer('zone_de_vegetation',['nature'],bbox=read_extent)
    veg=veg.loc[veg.nature.fillna('')!='Haie']
    acts=layer('zone_d_activite_ou_d_interet',['nature_detaillee'],bbox=read_extent)
    parks=acts.loc[acts.nature_detaillee.fillna('')=='Parc']
    circ=circuit_domain(); dom=domains()
    stations=pd.read_csv(DATA/'气象观测/Paris_WRF_validation/processed/station_inventory.csv',dtype={'station':str})
    sx,sy=PROJ.transform(stations.lon.to_numpy(),stations.lat.to_numpy())
    stations['x']=sx;stations['y']=sy
    stations.to_csv(CACHE/'meteorological_stations.csv',index=False)
    paris.to_file(CACHE/'D075.geojson',driver='GeoJSON')
    polygon_frame(circ).to_file(CACHE/'circuit_1125_domain.geojson',driver='GeoJSON')
    print('Rendering study-area panels',flush=True)
    fig=plt.figure(figsize=(183/25.4,111/25.4))
    gs=fig.add_gridspec(2,2,width_ratios=[1,1.95],left=.025,right=.985,bottom=.155,top=.935,wspace=.10,hspace=.26)
    a=fig.add_subplot(gs[0,0]);b=fig.add_subplot(gs[1,0]);c=fig.add_subplot(gs[:,1])
    # Panel a: geographic locator, with genuine WPS boundaries.
    ne=_release_path('C:/Users/12169/.local/share/cartopy/shapefiles/natural_earth')
    # Clip before reprojection to avoid distant, wrapped global polygons.
    europe=box(-25,32,25,65)
    land=gpd.read_file(ne/'physical/ne_110m_land.shp')
    land.geometry=land.geometry.intersection(europe)
    land=land.loc[~land.geometry.is_empty].to_crs(CRS)
    borders=gpd.read_file(ne/'cultural/ne_110m_admin_0_boundary_lines_land.shp')
    borders.geometry=borders.geometry.intersection(europe)
    borders=borders.loc[~borders.geometry.is_empty].to_crs(CRS)
    a.set_facecolor('#E9F2F4')
    draw_geo(a,land,facecolor='#F1EFE8',edgecolor='#B8C4C6',linewidth=.35)
    draw_geo(a,borders,color='#BCC2BE',linewidth=.35)
    colors=['#53666F','#16858B','#E18A16']
    for i,col in enumerate(colors,1):
        xy=dom[f'd0{i}'];a.plot(xy[:,0],xy[:,1],color=col,lw=.95,zorder=5)
    d1=dom['d01'];fit(a,(d1[:,0].min()-160000,d1[:,1].min()-70000,d1[:,0].max()+80000,d1[:,1].max()+60000))
    px,py=pgeom.representative_point().coords[0]
    a.scatter([px],[py],s=45,marker='*',facecolor='#C33232',edgecolor='white',linewidth=.35,zorder=8)
    a.annotate('Paris',(px,py),xytext=(0,6),textcoords='offset points',ha='center',va='bottom',fontsize=7,color='#B52D2D',zorder=9)
    for name,lon,lat in [('France',4.3,44.5),('UK',-2.5,53.2)]:
        xx,yy=PROJ.transform(lon,lat);a.text(xx,yy,name,ha='center',fontsize=7,color='#737C7A')
    panel(a,'a','WRF nesting')
    a.legend([Line2D([],[],color=col,lw=1) for col in colors],['d01  9 km','d02  3 km','d03  1 km'],loc='lower left',bbox_to_anchor=(.31,.025),fontsize=6.7,frameon=False,handlelength=1.3,labelspacing=.2,borderpad=.3)
    # Panel b: station support and representative Circuit domain, no wind results.
    b.set_facecolor('#FAFAF8')
    # Geographic DSM relief only; no WRF result or fabricated terrain.
    fit(b,(min(sx.min(),circ.bounds[0])-5000,min(sy.min(),circ.bounds[1])-3500,max(sx.max(),circ.bounds[2])+6500,max(sy.max(),circ.bounds[3])+5000))
    with rasterio.open(CACHE/'dem/paris_glo30_lamb93_50m.tif') as src:
        xl=b.get_xlim();yl=b.get_ylim()
        win=rasterio.windows.from_bounds(xl[0]-100,yl[0]-100,xl[1]+100,yl[1]+100,src.transform).round_offsets().round_lengths()
        dem=src.read(1,window=win)
        assert np.all(dem!=src.nodata), 'Incomplete elevation coverage'
        left,bottom,right,top=rasterio.windows.bounds(win,src.transform)
        dem_extent=(left,right,bottom,top)
    relief_cmap=LinearSegmentedColormap.from_list('Paris relief',
        ['#A8C7BD','#7FB39C','#9ABC83','#CFCE8A','#C6AA73','#B98B65'])
    relief=LightSource(azdeg=315,altdeg=45).shade(dem,cmap=relief_cmap,
        vmin=0,vmax=250,vert_exag=1,dx=50,dy=50,blend_mode='soft')
    # Display-only global contrast; elevation and slope geometry are unchanged.
    relief[:,:,:3]=np.clip(.5+1.12*(relief[:,:,:3]-.5),0,1)
    b.imshow(relief,extent=dem_extent,origin='upper',zorder=0,interpolation='bilinear')
    bwater=layer('surface_hydrographique')
    # Shared administrative edges only: never draw the clipped delivery edge
    # as if it were a department boundary.
    shared=[]
    for i in range(len(deps)):
        for j in range(i+1,len(deps)):
            edge=deps.geometry.iloc[i].boundary.intersection(deps.geometry.iloc[j].boundary)
            if not edge.is_empty: shared.append(edge)
    if shared:
        draw_geo(b,gpd.GeoDataFrame(geometry=shared,crs=CRS),color='#B8BDBF',linewidth=.35,zorder=2)
    draw_geo(b,bwater,facecolor=WATER,edgecolor='none',zorder=3)
    draw_geo(b,polygon_frame(circ),facecolor='none',edgecolor='#594078',linewidth=1.2,linestyle='--',zorder=5)
    draw_geo(b,paris,facecolor='none',edgecolor='#9C2F2B',linewidth=1.1,zorder=6)
    b.scatter(sx,sy,s=21,marker='^',facecolor='#123E62',edgecolor='white',linewidth=.55,zorder=8)
    offsets=[(-9,5),(-5,-12),(7,-3),(5,-6),(5,2),(5,-2),(5,1)]
    for i,(xx,yy,off) in enumerate(zip(sx,sy,offsets),1):
        b.annotate(str(i),(xx,yy),xytext=off,textcoords='offset points',fontsize=7,color='#102E45',zorder=10,path_effects=[pe.withStroke(linewidth=1.1,foreground='white')])
    fit(b,(min(sx.min(),circ.bounds[0])-5000,min(sy.min(),circ.bounds[1])-3500,max(sx.max(),circ.bounds[2])+6500,max(sy.max(),circ.bounds[3])+5000))
    panel(b,'b','Local computational setting')
    scale(b,10,y=.125)
    pos=b.get_position()
    ca=fig.add_axes([pos.x0+.065*pos.width,pos.y0+.88*pos.height,.37*pos.width,.022*pos.height])
    cb=fig.colorbar(matplotlib.cm.ScalarMappable(norm=Normalize(0,250),cmap=relief_cmap),cax=ca,orientation='horizontal',ticks=[0,125,250])
    cb.ax.tick_params(labelsize=6,length=2,pad=1)
    cb.outline.set_linewidth(.35)
    ca.set_title('Elevation (m)',fontsize=6.5,pad=3,color='black')
    # Panel c: true D075 boundary and the actual building footprint texture.
    c.set_facecolor('#F7F6F2')
    draw_geo(c,parks,facecolor=GREEN,edgecolor='none')
    draw_geo(c,veg,facecolor=GREEN,edgecolor='none')
    draw_geo(c,buildings,facecolor=BUILD,edgecolor='none')
    draw_geo(c,water,facecolor=WATER,edgecolor='none')
    draw_geo(c,paris,facecolor='none',edgecolor=CORE,linewidth=1.05)
    fit(c,extent)
    panel(c,'c','Paris: buildings and open spaces')
    north(c);scale(c,3,x=.07,y=.075)
    # Map labels identify geography, not model-derived hotspots.
    labels=[('Bois de\nBoulogne',2.246,48.860),('Bois de\nVincennes',2.442,48.828)]
    for name,lon,lat in labels:
        xx,yy=PROJ.transform(lon,lat)
        c.text(xx,yy,name,fontsize=8,ha='center',va='center',color='#355E42',path_effects=[pe.withStroke(linewidth=2,foreground='#F7F6F2')])
    xx,yy=PROJ.transform(2.320,48.863)
    c.annotate('Seine',(xx,yy),xytext=(4,17),textcoords='offset points',fontsize=8,color='#357890',arrowprops={'arrowstyle':'-','color':'#357890','lw':.6},path_effects=[pe.withStroke(linewidth=2,foreground='white')])
    # Compact legends and station key below the map, not over geographic texture.
    handles=[Patch(facecolor=BUILD,edgecolor='none',label='Buildings'),Patch(facecolor=GREEN,edgecolor='none',label='Woodland / parks'),Patch(facecolor=WATER,edgecolor='none',label='Water'),Line2D([],[],color=CORE,lw=1,label='D075 boundary')]
    c.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,-.02),ncol=2,frameon=False,columnspacing=1.7,handlelength=1.4,labelspacing=.65)
    fig.text(.027,.112,'Dashed outline: Circuit domain (112.5°)\nTriangles: meteorological stations',fontsize=6.7,color=INK,va='top',linespacing=1.4)
    fig.text(.027,.050,'1 Trappes   2 Toussus-le-Noble   3 Vélizy   4 Orly\n5 Le Bourget   6 Paris-Montsouris   7 Charles de Gaulle',fontsize=6.5,color=INK,va='top',linespacing=1.35)
    fig.canvas.draw()
    require_matplotlib_panel_alignment(fig,axes=[a,b,c],panel_ids=['a','b','c'],json_out=QA/'alignment.json',overlay_svg=QA/'alignment.svg',strict=True)
    stem='Fig00_study_area'
    fig.savefig(OUT/'pdf'/f'{stem}.pdf')
    fig.savefig(DEST/f'{stem}.png',dpi=300)
    shutil.copy2(DEST/f'{stem}.png',OUT/'png'/f'{stem}.png')
    # Byte-identical copies of the Python render avoid redundant rasterization.
    shutil.copy2(DEST/f'{stem}.png',QA/'preview.png')
    plt.close(fig)
    report={'role':'Study setting, not experimental evidence','building_features_loaded':len(buildings),'all_seven_stations':len(stations),'D075_area_km2':pgeom.area/1e6,'panels':{'a':'Actual-scale D075 administrative locator and three WPS mass-grid perimeters','b':'WPS 1 km elevation with interpolated contour display, shared department borders, water, seven weather stations, 112.5-degree Circuit domain','c':'Actual D075 boundary, all footprints in display extent, water, woodland and parks'},'no_result_fields_used':True,'sources':['IGN BD TOPO June 2022','WPS geogrid v4.7.0','NOAA ISD station inventory','Natural Earth 1:110m'],'display_only_exclusions':'Layer geometry is limited to map extent; vegetation excludes Haie consistently with the existing mask. No experimental observations are dropped.'}
    report['panels']['a']='Paris red-star locator and three WPS mass-grid perimeters'
    report['panels']['b']='Copernicus GLO-30 DSM resampled to 50 m Lambert-93; shaded relief (315-degree azimuth, 45-degree altitude, vertical exaggeration 1), 0–250 m hypsometric tint, BD TOPO water and shared boundaries, seven stations and Circuit domain'
    report['sources'].append('Copernicus GLO-30 public DSM, 2021 release via AWS')
    (QA/'source_manifest.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('Saved',DEST/f'{stem}.png',flush=True)

if __name__=='__main__': main()

"""Render the manuscript study-area map from the original geospatial layers."""
import os
from pathlib import Path
from datetime import datetime
import shutil
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LightSource
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon, Patch
from matplotlib import patheffects as pe
from pyproj import Transformer
ROOT = Path(__file__).resolve().parents[2]
CACHE = Path(os.environ.get("PARIS_MAP_CACHE", ROOT / "datas" / "external" / "study_area"))
DEST = ROOT / "outputs" / "figures"
DEST.mkdir(parents=True, exist_ok=True)
RED = "#C44F45"
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial","Liberation Sans","DejaVu Sans"], "legend.frameon": False})

def compass(ax, x=.94, y=.80, size=.13):
    """Split north arrow, corrected for Lambert-93 grid convergence at Paris."""
    tr=Transformer.from_crs(4326,2154,always_xy=True)
    px,py=tr.transform(2.35,48.8566); nx,ny=tr.transform(2.35,48.9566)
    angle=np.arctan2(nx-px,ny-py)
    p=ax.get_position(); ar=p.height*ax.figure.get_figheight()/(p.width*ax.figure.get_figwidth())
    size=min(size,5.0/(p.height*ax.figure.get_figheight()*25.4))
    def pts(values):
        v=np.array(values)*size
        xx=(v[:,0]*np.cos(angle)+v[:,1]*np.sin(angle))*ar
        yy=-v[:,0]*np.sin(angle)+v[:,1]*np.cos(angle)
        return np.c_[x+xx,y+yy]
    for points,fc in [([(0,1),(-.24,0),(0,.24)],'#202832'), ([(0,1),(.24,0),(0,.24)],'white')]:
        ax.add_patch(Polygon(pts(points),transform=ax.transAxes,facecolor=fc,
                            edgecolor='#202832',lw=.65,zorder=30))
    ax.text(x,y+size*1.13,'N',transform=ax.transAxes,ha='center',fontsize=7,zorder=31)


def graticule(ax, step, labels=True):
    """Genuine projected geographic graticules, never relabel kilometre axes."""
    from shapely.geometry import LineString, box
    tr=Transformer.from_crs(4326,2154,always_xy=True); inv=Transformer.from_crs(2154,4326,always_xy=True)
    x0,x1=ax.get_xlim();y0,y1=ax.get_ylim();lons,lats=inv.transform([x0,x1,x0,x1],[y0,y0,y1,y1])
    bbox=box(x0,y0,x1,y1);xt=[];xl=[];yt=[];yl=[]
    for lon in np.arange(np.floor(min(lons)/step)*step,max(lons)+step,step):
        lat=np.linspace(min(lats)-2,max(lats)+2,500);x,y=tr.transform(np.full(500,lon),lat)
        clipped=LineString(np.c_[x,y]).intersection(bbox)
        if clipped.geom_type=='LineString' and not clipped.is_empty:ax.plot(*clipped.xy,c='#6F7D86',lw=.35,alpha=.5,ls=':',zorder=25)
        hit=LineString(np.c_[x,y]).intersection(LineString([(x0,y0),(x1,y0)]))
        if hit.geom_type=='Point':xt.append(hit.x);xl.append(f'{abs(lon):g}°'+('E' if lon>=0 else 'W'))
    for lat in np.arange(np.floor(min(lats)/step)*step,max(lats)+step,step):
        lon=np.linspace(min(lons)-5,max(lons)+5,500);x,y=tr.transform(lon,np.full(500,lat))
        clipped=LineString(np.c_[x,y]).intersection(bbox)
        if clipped.geom_type=='LineString' and not clipped.is_empty:ax.plot(*clipped.xy,c='#6F7D86',lw=.35,alpha=.5,ls=':',zorder=25)
        hit=LineString(np.c_[x,y]).intersection(LineString([(x0,y0),(x0,y1)]))
        if hit.geom_type=='Point':yt.append(hit.y);yl.append(f'{lat:g}°N')
    ax.set_xlim(x0,x1);ax.set_ylim(y0,y1)
    if labels:
        ax.set_xticks(xt,xl,fontsize=6.5)
        ax.set_yticks(yt,yl,fontsize=6.5)
        ax.tick_params(length=2.5,width=.8,pad=1,colors='#303A42')


def save_current(fig, axes, stem):
    from matplotlib.lines import Line2D
    import numpy as np
    import geopandas as gpd
    from pyproj import Transformer
    from matplotlib.transforms import ScaledTranslation
    from matplotlib import patheffects as pe
    from shapely.geometry import box
    a,b,c = axes
    # Enlarge the already populated map, not its geographic bounds.
    pa,pb,pc=[ax.get_position().frozen() for ax in axes]
    full_height=pa.y1-pb.y0
    magnification=full_height/pc.height
    wide=1+pc.width*(magnification-1)
    fig.set_figwidth(fig.get_figwidth()*wide)
    for ax,p in ((a,pa),(b,pb)):
        ax.set_position([p.x0/wide,p.y0,p.width/wide,p.height])
    c.set_position([pc.x0/wide,pb.y0,pc.width*magnification/wide,full_height])
    for ax in axes:
        for spine in ax.spines.values():
            spine.set_color('#151515')
            spine.set_linewidth(1.15)
        ax.tick_params(labelsize=8, colors='#222222', pad=2)
        for txt in ax.texts:
            if txt.get_text() in ('(a)','(b)','(c)'):
                txt.set_position((0,1))
                txt.set_transform(ax.transAxes+ScaledTranslation(-3/72,3/72,fig.dpi_scale_trans))
                txt.set_ha('right')
                txt.set_va('bottom')
                txt.set_fontsize(10)
            else:
                txt.set_fontsize(txt.get_fontsize()+0.9)
    # Light cartographic base. Keep the domain outlines and locator on top.
    a.set_facecolor('#D8EAF2')
    a.collections[0].set_visible(False)
    a.collections[1].set_visible(False)
    region=box(-25,32,25,65)
    countries=gpd.read_file(CACHE/'ne_50m_admin_0_countries.zip',bbox=region.bounds)
    countries.geometry=countries.geometry.intersection(region)
    countries=countries[~countries.geometry.is_empty].to_crs(2154)
    fills={'FRA':'#E8DCC3','DEU':'#E2E7D9','ESP':'#F3E5CF','GBR':'#E4E7D8','BEL':'#E7DDE3','ITA':'#E4DDD3'}
    countries.plot(ax=a,color=countries.ADM0_A3.map(fills).fillna('#F0ECE3'),edgecolor='#8C9390',
                   linewidth=.40,aspect=None,zorder=1.1,autolim=False)
    tr=Transformer.from_crs(4326,2154,always_xy=True)
    for name,lon,lat in [('France',2.2,44.9),('Spain',-3.2,40.2),('Germany',8.9,51.2),('UK',-2.2,53.2),('Italy',13.35,42.5)]:
        x,y=tr.transform(lon,lat)
        if a.get_xlim()[0]<x<a.get_xlim()[1] and a.get_ylim()[0]<y<a.get_ylim()[1]:
            a.text(x,y,name,ha='center',va='center',fontsize=6.8,color='#4E5554',zorder=5,
                   path_effects=[pe.withStroke(linewidth=1.2,foreground='white')])
    for ax in axes:
        for line in ax.lines:
            if line.get_linestyle()==':':
                line.set_alpha(.27)
    c.set_facecolor('#FCFCFA')
    # Centre the domain key in the gutter, clear of the coordinate ticks.
    handles=[]
    for line,col,ls,width,label in zip(a.lines[:3],['#15518A','#C85F00','#7A2082'],
                                       ['-','--','-'],[1.35,1.4,1.7],['d01 9 km','d02 3 km','d03 1 km']):
        line.set(color=col,linestyle=ls,linewidth=width,zorder=7)
        handles.append(Line2D([],[],color=col,linestyle=ls,linewidth=width,label=label))
    a.get_legend().remove()
    key=a.legend(handles=handles, loc='upper center',bbox_to_anchor=(.50,-.13),
             ncol=3, fontsize=7.5,borderaxespad=0,frameon=False,
             handlelength=1.55, handletextpad=.3, columnspacing=.55, borderpad=.22)
    # Restore the earlier green-ochre hillshade and add a fine casing to both boundaries.
    im=b.images[0]
    rgb=np.array(im.get_array(),copy=True)
    rgb[...,:3]=.78*rgb[...,:3]+.22
    im.set_data(rgb)
    purple='#282064'
    for layer,color,style in zip(b.collections[:2],[purple,'#BB3B2B'],['--','-']):
        layer.set_edgecolor(color)
        layer.set_linewidth(1.25)
        layer.set_linestyle(style)
        layer.set_path_effects([pe.Stroke(linewidth=2.0,foreground='white'),pe.Normal()])
    for t in fig.texts: t.set_visible(False)
    b.legend(handles=[Line2D([],[],color=purple,ls='--',label='Circuit domain (112.5°)'),
                      Line2D([],[],marker='^',ls='none',color='#194D71',label='Weather station')],
             bbox_to_anchor=(-.01,-.13),loc='upper left',borderaxespad=0,ncol=2,
             fontsize=7.5,columnspacing=.7,handlelength=1.35,frameon=False)
    fig.text(.065/wide,.063,'1 Trappes   2 Toussus   3 Vélizy   4 Orly\n5 Le Bourget   6 Montsouris   7 Charles de Gaulle',
             fontsize=7.5,va='top',linespacing=1.25)
    for t in c.get_legend().get_texts(): t.set_fontsize(8.5)
    target = DEST / 'Fig00_study_area.png'
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    assert abs(c.bbox.y1-a.bbox.y1)<.1 and abs(c.bbox.y0-b.bbox.y0)<.1
    print('Panel c linear enlargement:',round(magnification,3),flush=True)
    legend_box=key.get_window_extent(renderer)
    if any(legend_box.overlaps(obj.get_window_extent(renderer)) for obj in [a,b,*a.get_xticklabels(),*b.texts]):
        raise ValueError('WRF legend intersects map or labels')
    if target.exists():
        backup = DEST / 'figure_backups' / datetime.now().strftime('%Y%m%d_%H%M%S')
        backup.mkdir(parents=True, exist_ok=True)
        shutil.copy2(target, backup / target.name)
        print('BACKUP', backup / target.name, flush=True)
    fig.savefig(target, dpi=300)
    plt.close(fig)
    print('SAVED', target, flush=True)


def fig02_study():
    import rasterio,pyogrio
    from shapely.geometry import box
    cache=CACHE
    paris=gpd.read_file(cache/'D075.geojson').to_crs(2154)
    circ=gpd.read_file(cache/'circuit_1125_domain.geojson').to_crs(2154)
    dom=np.load(cache/'wrf_domains.npz');stations=pd.read_csv(cache/'meteorological_stations.csv')
    raw=Path(os.environ['PARIS_BDTOPO_GPKG'])
    bounds=paris.total_bounds; bb=box(*bounds).buffer(1500).bounds
    def layer(name):
        columns={'zone_de_vegetation':['nature'],'zone_d_activite_ou_d_interet':['nature_detaillee']}.get(name,[])
        return pyogrio.read_dataframe(raw,layer=name,bbox=bb,columns=columns).to_crs(2154)
    fig=plt.figure(figsize=(190/25.4,120/25.4));gs=fig.add_gridspec(2,2,width_ratios=[1,1.87],left=.065,right=.985,bottom=.17,top=.95,wspace=.16,hspace=.27)
    a=fig.add_subplot(gs[0,0]);b=fig.add_subplot(gs[1,0]);c=fig.add_subplot(gs[:,1])
    def fit(ax,bounds):
        x0,y0,x1,y1=bounds;p=ax.get_position();ratio=p.width*fig.get_figwidth()/(p.height*fig.get_figheight());w=max(x1-x0,(y1-y0)*ratio);h=w/ratio
        ax.set(xlim=((x0+x1-w)/2,(x0+x1+w)/2),ylim=((y0+y1-h)/2,(y0+y1+h)/2));ax.set_aspect('equal',adjustable='datalim')
        for sp in ax.spines.values():sp.set_visible(True);sp.set_color('#56636C');sp.set_linewidth(.85)
    ne=Path(os.environ['PARIS_NATURAL_EARTH']);eu=box(-25,32,25,65)
    for file,style in [('physical/ne_110m_land.shp',dict(facecolor='#F0EEE5',edgecolor='#B1BABD',linewidth=.4)),('cultural/ne_110m_admin_0_boundary_lines_land.shp',dict(color='#B1BABD',linewidth=.4))]:
        g=gpd.read_file(ne/file);g.geometry=g.geometry.intersection(eu);g=g[~g.geometry.is_empty].to_crs(2154);g.plot(ax=a,aspect=None,rasterized=True,**style)
    colors=['#475866','#288F8D','#D78526']
    for i,col in enumerate(colors,1):a.plot(dom[f'd0{i}'][:,0],dom[f'd0{i}'][:,1],c=col,lw=1)
    xy=dom['d01'];fit(a,(xy[:,0].min()-100000,xy[:,1].min()-50000,xy[:,0].max()+80000,xy[:,1].max()+50000));a.set_facecolor('#E9F2F6')
    px,py=paris.geometry.iloc[0].representative_point().coords[0]
    a.scatter(px,py,marker='*',s=42,c=RED,edgecolors='white',lw=.3,zorder=30);a.annotate('Paris',(px,py),xytext=(0,-7),textcoords='offset points',ha='center',va='top',fontsize=7,c=RED,zorder=30)
    a.legend(handles=[Line2D([],[],c=col,label=f'd0{i} {res} km') for i,res,col in zip([1,2,3],[9,3,1],colors)],fontsize=5.5,loc='upper center',bbox_to_anchor=(.5,-.10),
             ncol=3,handlelength=1.0,columnspacing=.6,handletextpad=.35,frameon=False)
    sx,sy=stations.x.to_numpy(),stations.y.to_numpy();fit(b,(sx.min()-4500,sy.min()-4500,sx.max()+4500,sy.max()+4500))
    with rasterio.open(cache/'dem/paris_glo30_lamb93_50m.tif') as src:
        xl=b.get_xlim();yl=b.get_ylim();win=rasterio.windows.from_bounds(xl[0],yl[0],xl[1],yl[1],src.transform).round_offsets().round_lengths()
        dem=src.read(1,window=win);left,bottom,right,top=rasterio.windows.bounds(win,src.transform)
    cm=LinearSegmentedColormap.from_list('relief',['#98BDAE','#70A989','#A7BA78','#D5CE85','#B78B5E'])
    rgb=LightSource(315,45).shade(dem,cmap=cm,vmin=0,vmax=250,dx=50,dy=50,vert_exag=1,blend_mode='soft')
    b.imshow(rgb,extent=(left,right,bottom,top),origin='upper');circ.plot(ax=b,aspect=None,facecolor='none',edgecolor='#6D568D',linestyle='--',linewidth=1.1);paris.plot(ax=b,aspect=None,facecolor='none',edgecolor=RED,lw=1)
    b.scatter(sx,sy,marker='^',s=21,c='#194D71',edgecolors='white',lw=.5,zorder=30)
    for i,(x,y) in enumerate(zip(sx,sy),1):b.annotate(str(i),(x,y),xytext=(4,3),textcoords='offset points',fontsize=7,zorder=31,path_effects=[pe.withStroke(linewidth=1.5,foreground='white')])
    b.text(px,py+2300,'Paris',color=RED,fontsize=8,ha='center',zorder=31,path_effects=[pe.withStroke(linewidth=1.5,foreground='white')])
    c.set_facecolor('#F7F6F2')
    # Use a landscape frame around D075. The departmental delivery does not
    # cover distant suburbs; do not expand the view solely to fill a tall slot.
    p=c.get_position()
    focus=box(*bounds).buffer(480).bounds
    height=p.width*fig.get_figwidth()/fig.get_figheight()/1.48
    c.set_position([p.x0,p.y0+(p.height-height)/2,p.width,height])
    fit(c,focus)
    xl,yl=c.get_xlim(),c.get_ylim()
    bb=(xl[0],yl[0],xl[1],yl[1])
    buildings=layer('batiment');water=layer('surface_hydrographique');veg=layer('zone_de_vegetation')
    veg=veg[veg.nature.fillna('')!='Haie'];parks=layer('zone_d_activite_ou_d_interet');parks=parks[parks.nature_detaillee.fillna('')=='Parc']
    for g,col in [(parks,'#A7C39C'),(veg,'#A7C39C'),(buildings,'#939A9D'),(water,'#7CBBCD')]:g.plot(ax=c,aspect=None,facecolor=col,edgecolor='none',rasterized=True)
    paris.plot(ax=c,aspect=None,facecolor='none',edgecolor=RED,lw=1.1);fit(c,bb)
    c.text(px,py+1200,'Paris',color=RED,fontsize=10,ha='center',zorder=31,path_effects=[pe.withStroke(linewidth=2,foreground='white')])
    for ax,letter,step in [(a,'a',5),(b,'b',.2),(c,'c',.1)]:
        ax.text(-.10,1.035,f'({letter})',transform=ax.transAxes,weight='bold',fontsize=10)
        graticule(ax,step);compass(ax,x=.12 if letter=='b' else .93,y=.79,size=.12)
    for ax,km in [(b,10),(c,3)]:
        x0,x1=ax.get_xlim();y0,y1=ax.get_ylim();x=x0+(.18 if ax is b else .07)*(x1-x0);y=y0+.075*(y1-y0)
        ax.plot([x,x+km*1000],[y,y],c='#26333C',lw=1.3,zorder=32);ax.annotate(f'{km} km',(x+km*500,y),xytext=(0,3),textcoords='offset points',ha='center',fontsize=7,zorder=32)
    c.legend(handles=[Patch(color='#939A9D',label='Buildings'),Patch(color='#A7C39C',label='Woodland / parks'),Patch(color='#7CBBCD',label='Water'),Line2D([],[],c=RED,label='Paris boundary')],ncol=2,loc='upper center',bbox_to_anchor=(.5,-.06),fontsize=7)
    fig.text(.065,.11,'Dashed: Circuit domain (112.5°)\nTriangles: weather stations',fontsize=6.5,va='top')
    fig.text(.065,.05,'1 Trappes   2 Toussus   3 Vélizy   4 Orly\n5 Le Bourget   6 Montsouris   7 Charles de Gaulle',fontsize=6.5,va='top')
    save_current(fig,[a,b,c],'Fig02_study_area_revised')


if __name__ == "__main__":
    fig02_study()

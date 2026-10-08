"""Reproducible Paris Results figures from stored experiment outputs.

No synthetic data or subsampling. See FIGURE_PLAN.md and generated captions.
Run: python manuscript_figures/scripts/plot_results.py [2 3 ... 9]
"""

from release_paths import resolve_path as _release_path
from pathlib import Path
import sys
import json
import hashlib
import string
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / '.runtime'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import Normalize, TwoSlopeNorm, ListedColormap, LogNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.stats import spearmanr
from pyproj import Transformer
from sklearn.cluster import KMeans
from matplotlib.text import Text

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'manuscript_figures'
DATA = _release_path('E:/lunwen4/2')
MORPH = DATA / '建筑形态/Paris/processed/phase1a_v2'
STRICT = ROOT / 'new_study/17_transport_metric_candidates/analysis_strict_upwind_pbl'
SKILL = _release_path('C:/Users/12169/.codex/skills/nature-figure/scripts')
sys.path.insert(0, str(SKILL))
from release_plot_checks import require_matplotlib_panel_alignment

for folder in ('pdf', 'svg', 'png', 'tiff', 'source_data', 'qa'):
    (OUT / folder).mkdir(exist_ok=True, parents=True)

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial'],
    'mathtext.fontset': 'custom', 'mathtext.rm': 'Arial',
    'mathtext.it': 'Arial:italic', 'mathtext.bf': 'Arial:bold',
    'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 9,
    'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.linewidth': .65, 'xtick.major.width': .6, 'ytick.major.width': .6,
    'xtick.major.size': 3, 'ytick.major.size': 3,
    'lines.linewidth': 1.15, 'pdf.fonttype': 42, 'svg.fonttype': 'none',
    'savefig.facecolor': 'white', 'figure.facecolor': 'white',
    'legend.frameon': False, 'axes.unicode_minus': True,
})
TEAL, SLATE, OCHRE = '#167D8D', '#78838F', '#C58936'
BLUE, RED = '#32689B', '#BB5148'
DC = {45.: '#567FA3', 67.5: '#168777', 90.: '#C29535', 112.5: '#89578E'}
DIRS = list(DC)
META = {}
EXCLUSIONS = []
CAPTIONS = {}
ORIGIN = np.array([652000., 6862000.])
MAPLIM = (-11., 11., -8.4, 8.4)

def read_csv(path):
    path = Path(path)
    META[str(path)] = {'bytes': path.stat().st_size, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    return pd.read_csv(path)

def source(frame, name):
    if isinstance(frame, gpd.GeoDataFrame):
        frame = pd.DataFrame(frame.drop(columns='geometry'))
    frame.to_csv(OUT / 'source_data' / (name + '.csv'), index=False)

def axes_grid(rows=2, cols=2, height=151):
    six = rows*cols == 6
    height = 125 if six else 137
    fig, axes = plt.subplots(2, 3 if six else 2, figsize=(183/25.4, height/25.4))
    fig.subplots_adjust(left=.085 if six else .10, right=.975, bottom=.13, top=.91,
                        wspace=.57 if six else .42, hspace=.67)
    axes = np.asarray(axes).reshape(rows, cols)
    for i, ax in enumerate(axes.flat):
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
        ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
        ax.annotate('('+string.ascii_lowercase[i]+')', xy=(0, 1), xycoords='axes fraction',
                    xytext=(-20, 10), textcoords='offset points', weight='bold', fontsize=9,
                    annotation_clip=False)
    return fig, axes

def title(ax, txt):
    import textwrap
    ax.set_title('\n'.join(textwrap.wrap(txt, 26 if ax.figure.axes.__len__() >= 6 else 36)),
                 loc='left', pad=10, fontweight='normal', fontsize=8)

def zero(ax, axis='y'):
    (ax.axhline if axis == 'y' else ax.axvline)(0, color='#ACB4BC', lw=.65, ls='--', zorder=0)

def save(fig, axes, number, name, caption):
    stem = f'Fig{number:02d}_{name}'
    for text in fig.findobj(match=Text):
        s=text.get_text().replace('m s⁻¹','m/s').replace('µg m⁻³','µg/m³').replace('µmol m⁻³','µmol/m³')
        s=s.replace('×10⁻³','×0.001').replace('NO₂','NO$_2$').replace('O₃','O$_3$')
        text.set_text(s)
    fig.canvas.draw()
    require_matplotlib_panel_alignment(fig, axes=list(axes.flat),
        panel_ids=list(string.ascii_lowercase[:axes.size]),
        json_out=OUT/'qa'/f'{stem}.alignment.json',
        overlay_svg=OUT/'qa'/f'{stem}.alignment.svg', strict=True,
        tolerance_pt=1.5, gutter_tolerance_pt=1.5)
    fig.savefig(OUT/'pdf'/f'{stem}.pdf')
    # PDF is retained for geometry/font QA only; manuscript delivery is PNG.
    fig.savefig(OUT/'png'/f'{stem}.png', dpi=300)
    CAPTIONS[stem] = caption
    plt.close(fig)
    print('SAVED', stem, flush=True)

def date_axis(ax):
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d Jul'))
    ax.set_xlabel('2022 (UTC)')

def colourbar(fig, ax, artist, label, extend='neither'):
    # Explicit auxiliary axes preserve the equal scientific plot rectangles.
    p = ax.get_position()
    mm = fig.get_figheight()*25.4
    gap = 5.5 if ax.axison and len(ax.get_xticks()) else -2.0
    ca = fig.add_axes([p.x0, p.y0-(gap+1.8)/mm, p.width, 1.8/mm])
    cb = fig.colorbar(artist, cax=ca, orientation='horizontal', extend=extend)
    cb.ax.tick_params(labelsize=8, length=2, pad=1)
    cb.set_label(label, fontsize=8, labelpad=1)
    cb.locator = matplotlib.ticker.MaxNLocator(3)
    cb.update_ticks()
    return cb

def outline(ax):
    # Core-grid support polygon, not a newly inferred administrative boundary.
    for poly in getattr(CORE_GEOM, 'geoms', [CORE_GEOM]):
        xx, yy = poly.exterior.xy
        ax.plot((np.asarray(xx)-ORIGIN[0])/1000,
                (np.asarray(yy)-ORIGIN[1])/1000, color='#4D5962', lw=.5)

def map_frame(ax, scale=False):
    bounds=np.asarray(CORE_GEOM.bounds).reshape(2,2)
    center=(bounds.mean(axis=0)-ORIGIN)/1000
    span=(bounds[1]-bounds[0])/1000*1.06
    p=ax.get_position(); ratio=p.width*ax.figure.get_figwidth()/(p.height*ax.figure.get_figheight())
    width=max(span[0],span[1]*ratio); height=width/ratio
    ax.set_xlim(center[0]-width/2,center[0]+width/2)
    ax.set_ylim(center[1]-height/2,center[1]+height/2)
    ax.set_aspect('equal', adjustable='datalim')
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(False)
    outline(ax)
    if scale:
        x0,x1=ax.get_xlim(); y0,y1=ax.get_ylim()
        sy=y0+.12*(y1-y0)
        ax.plot([x0+.7,x0+5.7], [sy,sy], color='#36454F', lw=1.4)
        ax.annotate('5 km',xy=(x0+3.2,sy),xytext=(0,-5),textcoords='offset points',
                    ha='center',va='top',fontsize=8)
        ax.annotate('', xy=(x1-.8,y1-.8), xytext=(x1-.8,y1-2.4),
                    arrowprops={'arrowstyle':'->', 'lw':.8, 'color':'#36454F'})
        ax.text(x1-.8,y1-.4,'N',ha='center',fontsize=8)

def raster_map(ax, frame, values, norm, cmap='viridis', scale=False):
    xs = frame.x_center.to_numpy(float); ys = frame.y_center.to_numpy(float)
    xmin, ymin = xs.min()-50, ys.min()-50
    c = np.rint((xs-xs.min())/100).astype(int); r = np.rint((ys-ys.min())/100).astype(int)
    a = np.full((r.max()+1, c.max()+1), np.nan); a[r, c] = np.asarray(values)
    extent = [(xmin-ORIGIN[0])/1000, (xmin+100*a.shape[1]-ORIGIN[0])/1000,
              (ymin-ORIGIN[1])/1000, (ymin+100*a.shape[0]-ORIGIN[1])/1000]
    art = ax.imshow(a, origin='lower', extent=extent, interpolation='nearest', cmap=cmap, norm=norm)
    map_frame(ax, scale)
    return art

def response_map(ax, field, norm, selected):
    # Full saved WRF field, including the computational surroundings.
    a = ax.pcolormesh((WX-ORIGIN[0])/1000,(WY-ORIGIN[1])/1000,
                     np.ma.masked_invalid(field),cmap='RdBu_r',norm=norm,
                     shading='nearest',rasterized=True)
    x=(WX-ORIGIN[0])/1000; y=(WY-ORIGIN[1])/1000
    # REAL 10 m wind provides context only, not a layer-response vector.
    u=SURFACE['u10_earth'][selected].mean(axis=0)
    v=SURFACE['v10_earth'][selected].mean(axis=0)
    from pyproj import Geod
    lon2,lat2,_=Geod(ellps='WGS84').fwd(SURFACE['lon'],SURFACE['lat'],
        np.degrees(np.arctan2(u,v)),np.hypot(u,v)*100)
    x2,y2=Transformer.from_crs(4326,2154,always_xy=True).transform(lon2,lat2)
    vx=(x2-WX)/100; vy=(y2-WY)/100
    keep=(np.indices(x.shape)[0]%3==0)&(np.indices(x.shape)[1]%3==0)&(abs(x)<13)&(abs(y)<11)
    q=ax.quiver(x[keep],y[keep],vx[keep],vy[keep],color='#28343F',angles='xy',
        scale_units='xy',scale=2.0,width=.004,headwidth=3.5,zorder=3)
    ax.quiverkey(q,.76,1.05,3,'3 m/s',labelpos='E',coordinates='axes',fontproperties={'size':7})
    ax.set_xlim(-14,14); ax.set_ylim(-13,13)
    ax.set_aspect('equal',adjustable='datalim')
    ax.set_xticks([-10,0,10]); ax.set_yticks([-10,0,10]); ax.tick_params(labelsize=7,pad=1)
    for s in ax.spines.values(): s.set_visible(True)
    outline(ax)
    return a

def points_box(ax, arrays, labels, colors, ylabel):
    bp = ax.boxplot(arrays, positions=np.arange(len(arrays)), widths=.40, patch_artist=True,
                    showfliers=False, medianprops={'color':'#263A44','lw':1.2},
                    boxprops={'lw':.7}, whiskerprops={'lw':.7}, capprops={'lw':.7})
    for b,c in zip(bp['boxes'],colors): b.set_facecolor(c); b.set_alpha(.22)
    for i,(arr,c) in enumerate(zip(arrays,colors)):
        # Every replicate; deterministic offsets only avoid identical x positions.
        off=np.linspace(-.13,.13,len(arr))
        ax.scatter(i+off,arr,s=12,c=c,linewidths=.3,edgecolors='white',zorder=3)
    ax.set_xticks(range(len(labels)),labels); ax.set_ylabel(ylabel)

def fig2():
    fig, axs = axes_grid(3,2,224)
    # Keep two rows and three columns; wider panels use the existing page width.
    fig.subplots_adjust(left=.06,right=.985,wspace=.32,hspace=.67)
    for row,(field,label,vmax) in enumerate([('H','Building height (m)',50),('FAI','FAI at 112.5°',1.5)]):
        for col,case in enumerate(['REAL','HOMO']):
            ax=axs[row,col]; key=f'H_{case}_m' if field=='H' else f'FAI_{case}_1125'
            ar=raster_map(ax,MG,MG[key],Normalize(0,vmax),scale=(row==col==0))
            title(ax,f'{case} · '+('height' if field=='H' else 'directional FAI'))
            colourbar(fig,ax,ar,label,extend='max')
    ax=axs[2,0]; title(ax,'Joint morphology')
    hb=ax.hexbin(MG.LP_REAL_cond,MG.H_REAL_m,gridsize=30,mincnt=1,cmap='Greys',norm=LogNorm(),linewidths=0,rasterized=True)
    ax.scatter(TPL['LP_star_conditional'],TPL['H_star_m'],s=60,marker='D',c=TEAL,edgecolors='white',label='HOMO template')
    ax.set_xlabel('Urban-conditional plan fraction'); ax.set_ylabel('Building height (m)')
    ax.legend(loc='upper right'); ax.set_xlim(0,max(1.05,MG.LP_REAL_cond.max()*1.02))
    ax=axs[2,1]; title(ax,'Directional FAI distribution')
    ca=TPL['conservation_audit']
    pairs=[ca['building_footprint_proxy'],ca['building_volume_proxy'],ca['directional_frontal_area_proxy']['1125']]
    ratios=[]
    for p in pairs:
        keys=list(p); rv=p[next(k for k in keys if k.startswith('REAL'))]; hv=p[next(k for k in keys if k.startswith('HOMO'))]; ratios.append(hv/rv)
    values=np.sort(MG.lambda_f_1125_conditional.to_numpy())
    ax.plot(values,np.arange(1,len(values)+1)/len(values),color=TEAL,label='REAL')
    ax.axvline(TPL['directional_FAI_star_conditional']['lambda_f_1125'],color=OCHRE,ls='--',label='HOMO')
    ax.set_xlabel('Urban-conditional FAI'); ax.set_ylabel('Cumulative fraction'); ax.set_ylim(0,1.04)
    ax.legend(loc='lower right')
    source(MG,'Fig02_morphology'); source(pd.DataFrame({'quantity':['footprint','volume','frontal1125'],'ratio':ratios}),'Fig02_conservation')
    save(fig,axs,2,'morphology','REAL/HOMO isolates composite morphology heterogeneity while preserving aggregate building quantities. a–d, 100 m treatment-cell height and whole-cell directional FAI, with shared REAL/HOMO scales; extended colour bars mark values above the display maximum, with no cells removed. Blank cells are outside the treatment scope. e, all treatment cells in height–urban-conditional plan-fraction space; darker hexagons indicate more cells and the diamond is the joint HOMO template. f, empirical cumulative distribution of urban-conditional REAL FAI at 112.5° and the uniform HOMO reference (dashed line). Aggregate footprint, volume and directional frontal-area proxies are conserved. Outline: core-grid support; coordinates: Lambert-93. n = 8,229 treatment cells. These are 100 m morphology inputs, not final WRF UCP values.')

def fig3():
    fig,axs=axes_grid(3,2,224)
    norm=Normalize(0,float(np.quantile(CIR[['current_real','current_homo']],.99)))
    for ax,col,txt in zip(axs[0],['current_real','current_homo'],['REAL current','HOMO current']):
        ar=raster_map(ax,CIR,CIR[col],norm); title(ax,txt); colourbar(fig,ax,ar,'Current (model units)',extend='max')
    lim=float(np.quantile(abs(CIR.delta_current),.99))
    ar=raster_map(axs[1,0],CIR,CIR.delta_current,TwoSlopeNorm(0,-lim,lim),'RdBu_r',True)
    title(axs[1,0],'Network redistribution'); colourbar(fig,axs[1,0],ar,'Δcurrent: REAL − HOMO',extend='both')
    ax=axs[1,1]; title(ax,'Paired current distribution')
    hb=ax.hexbin(CIR.current_homo,CIR.current_real,gridsize=35,mincnt=1,cmap='viridis',norm=LogNorm(),linewidths=0,rasterized=True)
    mx=max(CIR.current_real.max(),CIR.current_homo.max())*1.04
    ax.plot([0,mx],[0,mx],c=SLATE,lw=.8,ls='--'); ax.set(xlim=(0,mx),ylim=(0,mx),xlabel='HOMO current',ylabel='REAL current')
    ax.text(.05,.91,f'ρ = {spearmanr(CIR.current_homo,CIR.current_real).statistic:.3f}',transform=ax.transAxes)
    ax=axs[2,0]; title(ax,'Retention of high-current cells')
    qs=[.1,.05,.01]; vals=[]
    for q in qs:
        n=int(np.ceil(len(CIR)*q)); a=set(np.argpartition(CIR.current_real.to_numpy(),-n)[-n:]); b=set(np.argpartition(CIR.current_homo.to_numpy(),-n)[-n:]); vals.append(100*len(a&b)/n)
    ax.bar(range(3),vals,color=TEAL,width=.55); ax.set_xticks(range(3),['Top 10%','Top 5%','Top 1%']); ax.set_ylabel('REAL/HOMO overlap (%)'); ax.set_ylim(0,105)
    ax=axs[2,1]; title(ax,'Resistance and network response')
    t=CIR.loc[CIR.delta_resistance.abs()>0]
    hb=ax.hexbin(t.delta_resistance,t.delta_current,gridsize=32,mincnt=1,cmap='viridis',norm=LogNorm(),linewidths=0,rasterized=True)
    ax.set_xlabel('Δresistance'); ax.set_ylabel('Δcurrent'); zero(ax); zero(ax,'x')
    source(CIR,'Fig03_current')
    save(fig,axs,3,'circuit','Detailed morphology redistributes current without replacing the entire high-current network. a,b, REAL/HOMO current under identical 112.5° inflow and electrical settings. c, signed difference; a–c colour limits use pooled 99th-percentile absolute values and saturation is explicitly marked. d, all paired core current values, with the identity line and Spearman correlation. e, intersection size divided by the fixed number of highest-current cells selected in each case. f, treated-cell resistance and current differences. Hexagon intensity represents cell count, not fitted probability. Core maps include all valid core cells; f includes only cells with nonzero resistance difference. Current is an electrical connectivity descriptor, not wind speed.')

def fig4():
    fig,axs=axes_grid(3,2,224)
    b=BG.copy(); t=pd.to_datetime(b.time_utc.str.replace('_',' '))
    ax=axs[0,0]; title(ax,'Common background inflow')
    ax.plot(t,b.inflow_from_deg,c='#B8C0C7',lw=.8)
    for d in DIRS:
        sel=(b.circuit_inflow_bin_deg==d)&(b.speed_ms>=1)&(b.persistence6h>=.9)&(t>=pd.Timestamp('2022-07-17'))
        ax.scatter(t[sel],b.inflow_from_deg[sel],s=12,color=DC[d],label=f'{d:g}°')
    ax.set_ylabel('Wind from (°)'); ax.set_ylim(0,360); ax.set_yticks([0,90,180,270,360]); date_axis(ax)
    ax.legend(ncol=2,loc='upper left',handletextpad=.2,columnspacing=.6)
    ax=axs[0,1]; title(ax,'Background wind speed')
    ax.plot(t,b.speed_ms,c=TEAL); ax.axhline(1,c=SLATE,lw=.7,ls='--'); ax.set_ylabel('Wind speed (m s⁻¹)'); date_axis(ax)
    for ax,var,lab in [(axs[1,0],'temp_c','Temperature (°C)'),(axs[1,1],'wind_speed_ms','Wind speed (m s⁻¹)')]:
        title(ax,'Surface '+('temperature' if var=='temp_c' else 'wind'))
        z=SURF.groupby('time_utc')[[f'obs_{var}',f'wrf_{var}']].mean()
        ax.plot(z.index,z[f'obs_{var}'],color='#384958',label='Observed',marker='o',markersize=2,markevery=6)
        ax.plot(z.index,z[f'wrf_{var}'],color=TEAL,ls='--',label='WRF REAL')
        ax.set_ylabel(lab); date_axis(ax); ax.legend(loc='upper left')
        if var=='temp_c': ax.set_ylim(10,50)
    names=['Trappes','Toussus','Vélizy','Orly','Le Bourget','Montsouris','CDG']
    assert len(names)==len(SM)
    for ax,col,lab in [(axs[2,0],'temp_c_bias','Temperature bias (°C)'),(axs[2,1],'wind_speed_ms_rmse','Wind RMSE (m s⁻¹)')]:
        title(ax,'Station-level '+('cold bias' if 'bias' in col else 'wind error'))
        ax.scatter(SM[col],np.arange(len(SM)),color=TEAL,s=27)
        ax.set_yticks(np.arange(len(SM)),names,fontsize=7); ax.set_xlabel(lab); ax.invert_yaxis(); zero(ax,'x')
    source(SURF,'Fig04_surface_pairs'); source(BG,'Fig04_background'); source(SM,'Fig04_station_metrics')
    save(fig,axs,4,'weather_validation','The simulation captures the event evolution while retaining surface cold bias and station-dependent wind error. a, common d02 background inflow; coloured points are the four stable post-spinup direction bins used by the network analysis. b, background wind speed and the 1 m/s screening threshold. c,d, hourly means across available matched station observations and WRF values after spinup; no uncertainty band is inferred from station heterogeneity. e,f, station-level temperature bias and wind-speed RMSE. Station-specific missing values follow the stored observation QA. All times are UTC. Continuous background traces include spinup for context; analysis starts 17 July.')

def fig5():
    fig,axs=axes_grid(3,2,224)
    norm=TwoSlopeNorm(0,-.08,.08)
    for ax,d in zip(axs[:2].flat,DIRS):
        y=MODEL[f'y_10_1000_{int(d*10):04d}']
        selected=POST&(TM['background_bin_deg']==d)&(TM['background_speed_ms']>=1)&(TM['background_persistence6h']>=.9)
        field=TM['delta_along_10_1000'][selected].mean(axis=0)
        assert np.allclose(field[ACTIVE],y,atol=1e-7)
        ar=response_map(ax,field,norm,selected); title(ax,f'{d:g}° inflow'); colourbar(fig,ax,ar,'Δlayer velocity (m/s)',extend='both')
    ax=axs[2,0]; title(ax,'Hourly core response')
    vals=TM['delta_along_10_1000'][:,ACTIVE]; ts=TIMES[POST]
    lo,med,hi=np.quantile(vals[POST],[.25,.5,.75],axis=1)
    ax.fill_between(ts,lo,hi,color=TEAL,alpha=.18,label='Spatial IQR')
    ax.plot(ts,np.mean(vals[POST],axis=1),c=TEAL,label='Spatial mean'); zero(ax); date_axis(ax)
    ax.set_ylabel('Δvelocity (m s⁻¹)'); ax.set_ylim(-.03,.07); ax.legend(loc='upper right')
    ax=axs[2,1]; title(ax,'Opposing local responses')
    arr=[MODEL[f'y_10_1000_{int(d*10):04d}'] for d in DIRS]
    points_box(ax,arr,[f'{d:g}°' for d in DIRS],[DC[d] for d in DIRS],'Δvelocity (m s⁻¹)'); zero(ax)
    source(pd.DataFrame({'time_utc':ts,'mean':np.mean(vals[POST],axis=1),'q25':lo,'q75':hi}),'Fig05_hourly')
    source(pd.DataFrame({str(d):v for d,v in zip(DIRS,arr)}),'Fig05_directional_response')
    save(fig,axs,5,'transport_response','REAL morphology produces positive and negative low-layer velocity responses within the Paris core. a–d, direction-conditioned temporal means of REAL minus HOMO along-wind velocity averaged over 10–1000 m; all maps use the same zero-centred colour scale and the actual WRF grid centres. e, hourly mean and spatial interquartile range across the 85 treatment cells after spinup; the band describes spatial spread, not confidence. f, all 85 direction-mean cell values; boxes show median and IQR, whiskers extend to 1.5 IQR, and all cells are overlaid. Positive values indicate a larger component along the common background transport direction in REAL. This is a velocity proxy, not a volume or pollutant flux.')

def fig6():
    fig,axs=axes_grid(2,2,155)
    ax=axs[0,0]; title(ax,'Spatial validation blocks')
    blocks=KMeans(n_clusters=5,random_state=0,n_init=20).fit_predict(XY)
    # Same deterministic seed-0 partition as the stored validation experiment.
    colors=['#567FA3','#168777','#C29535','#89578E','#BB675D']
    field=np.full(ACTIVE.shape,np.nan); field[ACTIVE]=blocks
    ax.pcolormesh((WX-ORIGIN[0])/1000,(WY-ORIGIN[1])/1000,
                  np.ma.masked_invalid(field),cmap=ListedColormap(colors),
                  norm=Normalize(-.5,4.5),shading='nearest',rasterized=True)
    map_frame(ax)
    ax.legend(handles=[Line2D([],[],marker='s',ls='',color=c,label=str(i+1)) for i,c in enumerate(colors)],
              title='Held-out block',ncol=5,loc='lower center',bbox_to_anchor=(.5,-.23),
              handletextpad=.2,columnspacing=.7,fontsize=7,title_fontsize=8)
    source(pd.DataFrame({'x':XY[:,0],'y':XY[:,1],'seed0_block':blocks+1}),'Fig06_spatial_blocks')
    ax=axs[0,1]; title(ax,'Paired held-out errors')
    q=TEST.loc[TEST.metric=='10_1000']
    for d in DIRS:
        z=q[q.direction==d]
        for _,r in z.iterrows():
            ax.plot([0,1],[r.rmse_baseline,r.rmse_circuit],c=DC[d],alpha=.5,lw=.7)
        ax.scatter(np.zeros(len(z)),z.rmse_baseline,c=DC[d],s=10,marker='o')
        ax.scatter(np.ones(len(z)),z.rmse_circuit,c=DC[d],s=13,marker='D',label=f'{d:g}°')
    ax.set_xticks([0,1],['Local + context','+ Circuit']); ax.set_xlim(-.2,1.2); ax.set_ylim(.012,.027); ax.set_ylabel('CV RMSE (m s⁻¹)')
    ax.legend(ncol=2,loc='upper right',fontsize=7)
    ax=axs[1,0]; title(ax,'Gain depends on direction')
    points_box(ax,[-100*q[q.direction==d].relative_rmse_change.to_numpy() for d in DIRS],[f'{d:g}°' for d in DIRS],[DC[d] for d in DIRS],'RMSE reduction (%)'); zero(ax)
    ax=axs[1,1]; title(ax,'Network–response association')
    x=MODEL['network_1125'].ravel(); y=MODEL['y_10_1000_1125']
    ax.scatter(x*1000,y,s=19,c=TEAL,alpha=.75,edgecolors='white',lw=.3)
    ax.set_xlabel('Δcurrent (×10⁻³; 112.5°)'); ax.set_ylabel('Δvelocity (m s⁻¹)'); zero(ax); zero(ax,'x')
    source(q,'Fig06_cv'); source(pd.DataFrame({'delta_current':x,'delta_velocity':y}),'Fig06_joint')
    save(fig,axs,6,'network_added_value','Circuit improves spatial prediction beyond the specified local and context morphology baseline. a, the five spatial validation blocks for seed 0, reconstructed with the original KMeans settings; each block is held out in turn. The other nine partitions are included in b,c. Context predictors use available core cells within a 3.5 km neighbourhood and a 5 km upwind sector. b, paired baseline and Circuit-extended held-out RMSE; each line is one repeated five-block spatial partition. c, relative RMSE reduction across all ten partitions per direction (positive means improvement); boxes show median/IQR and 1.5-IQR whiskers with every partition overlaid. d, all 85 core cells for the 112.5° mean-map association; this raw association is not the added-value test. There are four direction-conditioned targets and 40 comparisons, not 40 independent meteorological events.')

def fig7():
    fig,axs=axes_grid(2,2,155)
    mets=['10_500','10_800','10_1000','pbl_mean','pbl_int']; labels=['10–500 m','10–800 m','10–1000 m','PBL mean','PBL integral']
    vals=np.array([[-100*TEST[(TEST.metric==m)&(TEST.direction==d)].relative_rmse_change.median() for d in DIRS] for m in mets])
    ax=axs[0,0]; title(ax,'Vertical and directional dependence')
    ar=ax.imshow(vals,cmap='RdBu',norm=TwoSlopeNorm(0,-12,12),aspect='auto')
    ax.set_yticks(range(5),labels,fontsize=7); ax.set_xticks(range(4),[f'{d:g}°' for d in DIRS])
    for i in range(5):
        for j in range(4): ax.text(j,i,f'{vals[i,j]:.1f}',ha='center',va='center',fontsize=8,color='white' if abs(vals[i,j])>7 else '#273440')
    colourbar(fig,ax,ar,'Median RMSE reduction (%)',extend='both')
    ax=axs[0,1]; title(ax,'Sensitivity across partitions')
    arrays=[-100*TEST[TEST.metric==m].relative_rmse_change.to_numpy() for m in mets]
    bp=ax.boxplot(arrays,vert=False,patch_artist=True,showfliers=True,widths=.5,medianprops={'c':TEAL},flierprops={'markersize':2,'markeredgecolor':SLATE})
    for p in bp['boxes']: p.set_facecolor('#D8E9EA')
    ax.set_yticks(range(1,6),labels); ax.invert_yaxis(); ax.set_xlabel('RMSE reduction (%)'); zero(ax,'x')
    ax=axs[1,0]; title(ax,'Network-label permutation')
    null=np.load(STRICT/'permutation_nav_null.npz')['combined']*100
    pj=json.loads((STRICT/'permutation_nav_results.json').read_text())
    obs=pj['combined_observed_mean_relative_rmse_change']*100
    ax.hist(null,bins=15,color='#C5CFD8',edgecolor='white'); ax.axvline(obs,c=TEAL,lw=1.6)
    ax.set_xlabel('Mean RMSE change (%)'); ax.set_ylabel('Permutations')
    ax.text(.97,.94,f"Observed {obs:.2f}%\np = {pj['combined_one_sided_p']:.4f}",ha='right',va='top',transform=ax.transAxes)
    ax=axs[1,1]; title(ax,'Unequal event support')
    counts=[int(TEST[TEST.direction==d].n_hours.iloc[0]) for d in DIRS]
    ax.bar(range(4),counts,color=[DC[d] for d in DIRS],width=.6)
    ax.set_xticks(range(4),[f'{d:g}°' for d in DIRS]); ax.set_ylabel('Matched hourly states'); ax.set_ylim(0,34)
    source(TEST,'Fig07_all_metric_tests'); source(pd.DataFrame({'null_mean_rmse_change_pct':null}),'Fig07_permutation')
    save(fig,axs,7,'vertical_direction_tests','Network gains are strongest and most consistent for the deeper fixed layers, but evidence is uneven across directions. a, median relative RMSE reduction by target and direction; the sign is positive for improvement. b, distributions of all 40 partition comparisons per target (median, IQR, 1.5-IQR whiskers and outliers); these are split sensitivities, not independent-event confidence intervals. c, the separate 100-label-permutation diagnostic with its observed mean RMSE change (negative means improvement in this panel, as labelled). d, hourly sample counts supporting the four direction-mean targets. PBL denotes the model boundary layer; the integral and mean are separate target definitions.')

def fig8():
    fig,axs=axes_grid(2,2,158)
    ax=axs[0,0]; title(ax,'Local climate zones')
    cs=LCZ.loc[LCZ.analysis_core.astype(bool)].copy(); cats=sorted(cs.lcz_int.unique())
    palette=['#9F3E32','#CF5D48','#E4967B','#A77593','#C9A0B5','#E0BCCB','#BAAF95','#CFB486','#ACA9A3','#645956','#3D7656','#91B184','#C1D09F','#DBDFB7','#B8C3C9','#C0A479','#75AFC9']
    codes=np.array([cats.index(c) for c in cs.lcz_int]); ar=raster_map(ax,cs,codes,Normalize(-.5,len(cats)-.5),ListedColormap([palette[int(c)-1] for c in cats]))
    handles=[Line2D([],[],marker='s',ls='',color=palette[int(c)-1],label=str(c) if c<=10 else chr(65+int(c)-11),markersize=5) for c in cats]
    ax.legend(handles=handles,ncol=5,loc='lower center',bbox_to_anchor=(.5,-.28),handletextpad=.1,columnspacing=.6,fontsize=7)
    ax=axs[0,1]; title(ax,'Class-based versus detailed FAI')
    ca=read_csv(DATA/'LCZ/Paris/Cerema_2022/circuit_benchmark_v1/1125/cells.csv'); ca=ca[ca.analysis_core.astype(bool)]
    hb=ax.hexbin(ca.fai_real,ca.fai_lcz,gridsize=30,mincnt=1,cmap='viridis',norm=LogNorm(),linewidths=0,rasterized=True)
    ax.set_xlabel('Detailed 3D FAI'); ax.set_ylabel('LCZ FAI (112.5°)')
    ax=axs[1,0]; title(ax,'Directional network gains')
    for j,d in enumerate(DIRS):
        z=LTEST[LTEST.direction==d]
        for _,r in z.iterrows(): ax.plot([j-.14,j+.14],[-100*r.relative_change_lcz,-100*r.relative_change_3d],c='#CDD2D8',lw=.6)
        ax.scatter(np.repeat(j-.14,len(z)),-100*z.relative_change_lcz,s=16,c=OCHRE,marker='s',label='LCZ' if j==0 else None)
        ax.scatter(np.repeat(j+.14,len(z)),-100*z.relative_change_3d,s=16,c=TEAL,marker='o',label='Detailed 3D' if j==0 else None)
    ax.set_xticks(range(4),[f'{d:g}°' for d in DIRS]); ax.set_ylabel('RMSE reduction (%)'); ax.set_ylim(-2,14); zero(ax); ax.legend(loc='upper left')
    ax=axs[1,1]; title(ax,'Across all 40 comparisons')
    points_box(ax,[-100*LTEST.relative_change_lcz.to_numpy(),-100*LTEST.relative_change_3d.to_numpy()],['LCZ','Detailed 3D'],[OCHRE,TEAL],'RMSE reduction (%)'); zero(ax)
    source(ca,'Fig08_fai_comparison'); source(LTEST,'Fig08_lcz_cv'); source(cs,'Fig08_lcz_cells')
    save(fig,axs,8,'lcz_benchmark','Locally calibrated LCZ preserves some network information but detailed 3D is more consistently beneficial across the tested directions. a, core LCZ classes (1–10 built types, A–G natural types). b, detailed and LCZ-derived FAI at 112.5° for all core cells; hexagons encode counts. c, paired LCZ and detailed-3D RMSE reductions for the same ten spatial partitions in each direction. d, all 40 comparisons; boxes show median/IQR with 1.5-IQR whiskers and every comparison overlaid. Positive reduction indicates benefit over the identical local/context baseline. LCZ class FAI is calibrated from local detailed morphology; this is a Circuit benchmark, not an LCZ–WRF simulation. The 90° advantage of LCZ is retained.')

def demean(z,key):
    return z[key]-z.groupby('station_code')[key].transform('mean')-z.groupby('time_utc')[key].transform('mean')+z[key].mean()

def fig9():
    fig,axs=axes_grid(3,2,224)
    aqbase=DATA/'空气质量/Paris_2022_event'; pairs=read_csv(aqbase/'analysis/D075_background_observation_wrf_pairs.csv')
    st=read_csv(aqbase/'analysis/D075_station_wrf_grid_match.csv')
    res=json.loads((aqbase/'analysis/air_quality_results.json').read_text())['contemporaneous_double_demeaned_results']
    ax=axs[0,0]; title(ax,'Urban-background stations')
    map_frame(ax,True)
    xx,yy=Transformer.from_crs(4326,2154,always_xy=True).transform(st.longitude,st.latitude)
    ozone=set(pairs.loc[pairs.O3.notna(),'station_code'])
    for x,y,code in zip(xx,yy,st.station_code):
        x=(x-ORIGIN[0])/1000; y=(y-ORIGIN[1])/1000
        ax.scatter(x,y,marker='D' if code in ozone else 'o',s=35,c=TEAL,edgecolors='white',lw=.5)
        offset,ha={'PA13':((-5,-14),'left'),'PA18':((-8,10),'right'),
                   'PA15L':((-5,-10),'right'),'PA12':((10,9),'left'),
                   'PA07':((3,-10),'left'),'PA01H':((-1,6),'center')}[code]
        ax.annotate(code,(x,y),xytext=offset,textcoords='offset points',fontsize=7,ha=ha)
    ax.legend(handles=[Line2D([],[],ls='',marker='D',color=TEAL,label='NO₂ / O₃'),Line2D([],[],ls='',marker='o',color=TEAL,label='NO₂ only')],loc='lower center',bbox_to_anchor=(.5,-.16),fontsize=8,ncol=2,handletextpad=.3,columnspacing=.7)
    ax=axs[0,1]; title(ax,'Ozone during the event')
    for code,col,ls in zip(sorted(ozone),[TEAL,OCHRE,SLATE],['-','--',':']):
        z=pairs[pairs.station_code==code]; ax.plot(pd.to_datetime(z.time_utc),z.O3,label=code,color=col,ls=ls)
    ax.set_ylabel('O₃ (µg m⁻³)'); ax.set_ylim(0,200); date_axis(ax); ax.legend(ncol=3,fontsize=7,loc='upper left',columnspacing=.7,handlelength=1.2)
    formal=pairs[pairs.background_bin_deg.isin(DIRS)]
    configs=[('NO2','NO₂','µg m⁻³'),('O3','O₃','µg m⁻³'),('Ox_umol_m3','Ox','µmol m⁻³')]
    for ax,(key,label,unit) in zip([axs[1,0],axs[1,1],axs[2,0]],configs):
        z=formal.dropna(subset=[key,'delta_transport']).copy(); x=demean(z,'delta_transport'); y=demean(z,key)
        EXCLUSIONS.append({'panel':label,'before':len(formal),'after':len(z),'rule':'nonmissing pollutant and delta_transport in formal four direction bins'})
        ax.scatter(x,y,s=10,color=TEAL,alpha=.48,lw=0,rasterized=True)
        zero(ax); zero(ax,'x'); title(ax,f'{label} association · n = {len(z)}')
        ax.set_xlabel('Δvelocity residual (m s⁻¹)'); ax.set_ylabel(f'{label} residual ({unit})')
        source(pd.DataFrame({'station':z.station_code,'time_utc':z.time_utc,'x_residual':x,'y_residual':y}),f'Fig09_{label}_residuals')
    ax=axs[2,1]; title(ax,'Association and uncertainty')
    for i,r in enumerate(res):
        v=r['spearman_rho_double_demeaned']; lo=r['block_bootstrap_ci_low']; hi=r['block_bootstrap_ci_high']
        ax.errorbar(v,i,xerr=[[v-lo],[hi-v]],fmt='o',color=TEAL,capsize=3,markersize=5,lw=1.2)
    ax.set_yticks(range(3),['NO$_2$','O$_3$','Ox']); ax.invert_yaxis(); ax.set_xlim(-.8,.4); ax.set_xlabel('Spearman ρ (95% block CI)'); zero(ax,'x')
    source(pairs,'Fig09_observation_pairs'); source(pd.DataFrame(res),'Fig09_intervals')
    save(fig,axs,9,'air_quality','Air-quality associations are pollutant-specific rather than a uniform pollution reduction. a, six urban-background stations; diamonds identify the three sites with paired O3/NO2 for Ox. b, observed O3 over the post-spinup event window. c–e, all valid station-hour pairs in the 58 direction-matched states after removal of station and common-hour means; no regression line is imposed. f, stored Spearman associations and 95% intervals from the original 6-hour moving-block bootstrap. NO2 has 345 pairs and O3/Ox each 139. Ox is the molar sum of O3 and NO2; a positive rescaling to O3-equivalent mass in the original statistics leaves ranks unchanged. Neither NO2 nor Ox has an interval excluding zero. Predictor: simulated REAL–HOMO layer-mean velocity response, not observed counterfactual pollution.')

if __name__ == '__main__':
    # All source arrays are read-only; each output has its own source-data table.
    mask=gpd.read_file(MORPH/'urban_mask_100m_v1/paris_urban_support_mask_100m_v1.gpkg')
    CORE_GEOM=mask.geometry.union_all()
    MG=gpd.read_file(MORPH/'real_homo_design_v2/urban_cells_reference.gpkg')
    MG['x_center']=MG.geometry.centroid.x; MG['y_center']=MG.geometry.centroid.y
    TPL=json.loads((MORPH/'real_homo_design_v2/homo_template.json').read_text())
    CIR=read_csv(MORPH/'extended_domain_v1/circuit_real_homo_1125_v1/analysis/circuit_real_homo_cellwise.csv')
    CIR=CIR.loc[CIR.analysis_core.astype(bool)&(CIR.current_real>-9000)&(CIR.current_homo>-9000)].copy()
    TEST=read_csv(STRICT/'strict_upwind_all_tests.csv'); MODEL=np.load(STRICT/'strict_model_inputs.npz'); XY=MODEL['xy_1125']
    TM=np.load(ROOT/'new_study/17_transport_metric_candidates/paired_transport_metrics.npz')
    TIMES=pd.to_datetime([str(t).replace('_',' ') for t in TM['times']]); POST=TIMES>=pd.Timestamp('2022-07-17')
    template=np.load(MORPH/'wrf_activation_pilot_v1/injection_templates/paris_d03_real_homo_injection_templates.npz')
    ACTIVE=template['PRIMARY_ACTIVATION'].astype(bool)&template['TREATMENT_CORE'].astype(bool)
    wrfgrid=np.load(ROOT/'new_study/15_network_added_value/phase1_1125/wrf_phase1_export.npz')
    WX,WY=Transformer.from_crs(4326,2154,always_xy=True).transform(wrfgrid['lon'],wrfgrid['lat'])
    assert np.allclose(XY,np.column_stack([WX[ACTIVE],WY[ACTIVE]]))
    BG=read_csv(ROOT/'new_study/13_wrf_homo_pair/full96h_analysis/d02_background_wind_hourly.csv')
    VD=DATA/'气象观测/Paris_WRF_validation/processed'
    SURFACE=np.load(VD/'wrf_real_surface_97h.npz')
    assert list(SURFACE['times'])==list(TM['times'])
    SURF=read_csv(VD/'wrf_isd_hourly_pairs.csv'); SURF['time_utc']=pd.to_datetime(SURF.time_utc); SURF=SURF[SURF.time_utc>=pd.Timestamp('2022-07-17')]
    SM=read_csv(VD/'surface_validation_by_station.csv')
    LCZ=gpd.read_file(DATA/'LCZ/Paris/Cerema_2022/processed_100m/paris_cerema_lcz_100m.gpkg')
    LTEST=read_csv(DATA/'LCZ/Paris/Cerema_2022/circuit_benchmark_v1/analysis/lcz_vs_3d_nav_all_folds.csv')
    selected=[int(a) for a in sys.argv[1:]] or list(range(2,10))
    for n in selected: globals()[f'fig{n}']()
    cp=OUT/'captions.json'; old=json.loads(cp.read_text(encoding='utf-8')) if cp.exists() else {}; old.update(CAPTIONS)
    cp.write_text(json.dumps(old,indent=2,ensure_ascii=False),encoding='utf-8')
    (OUT/'FIGURE_CAPTIONS.md').write_text('# Results figure legends\n\n'+'\n\n'.join(f'## {k}\n\n{v}' for k,v in sorted(old.items())),encoding='utf-8')
    (OUT/'qa/source_manifest.json').write_text(json.dumps(META,ensure_ascii=False,indent=2),encoding='utf-8')
    (OUT/'qa/exclusions.json').write_text(json.dumps(EXCLUSIONS,indent=2),encoding='utf-8')

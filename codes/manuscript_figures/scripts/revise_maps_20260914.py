"""Presentation-only revision of the three requested maps; original outputs stay intact.

Python backend; Arial; 300 dpi PNG delivery and internal editable PDF QA.
The original observations, masks, temporal means and response scale are retained.
Run with the existing scientific Python environment. Optional arguments: study morphology wrf.
"""
from pathlib import Path
import sys
import json
import shutil
import string

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import plot_results as r
import plot_study_area as s
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm, LogNorm
from matplotlib.patches import Polygon
from matplotlib.lines import Line2D
from pyproj import Transformer, Geod
import cartopy.crs as ccrs
import matplotlib.ticker as mticker

OUT = r.ROOT / 'manuscript_figures/revision_20260914'
for folder in ('png', 'pdf', 'qa', 'source_data'):
    (OUT / folder).mkdir(parents=True, exist_ok=True)
r.OUT = OUT
mpl.rcParams.update({'font.family': 'Arial', 'font.size': 8, 'axes.titlesize': 8,
    'axes.labelsize': 7.5, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
    'legend.fontsize': 7, 'pdf.fonttype': 42, 'svg.fonttype': 'none',
    'axes.titlecolor': 'black', 'text.color': 'black', 'axes.labelcolor': 'black'})

HEIGHT = LinearSegmentedColormap.from_list('Height copper',
    ['#FFF7ED', '#EFC49F', '#D38757', '#AA4C35', '#702A30', '#3C1428'])
FAI = LinearSegmentedColormap.from_list('FAI teal',
    ['#EFF7F3', '#BBDDCF', '#65B4AD', '#258B96', '#175A79', '#153547'])


def label(ax, letter, text=''):
    ax.annotate(f'({letter})', xy=(0, 1), xycoords='axes fraction',
        xytext=(0, 7), textcoords='offset points', fontsize=9,
        fontweight='bold', color='black', annotation_clip=False)
    if text:
        ax.annotate(text, xy=(0, 1), xycoords='axes fraction',
            xytext=(18, 7), textcoords='offset points', fontsize=8,
            color='black', annotation_clip=False)


def compass(ax, x=.942, y=.85, size=.092):
    """Split black/white geographic north arrow, calibrated to Lambert-93."""
    lon, lat = 2.35, 48.86
    trans = Transformer.from_crs(4326, 2154, always_xy=True)
    x0, y0 = trans.transform(lon, lat)
    xn, yn = trans.transform(lon, lat + .02)
    # Positions in axes coordinates, with physical correction for panel aspect.
    p = ax.get_position()
    ratio = p.width * ax.figure.get_figwidth() / (p.height * ax.figure.get_figheight())
    ang = np.arctan2(xn-x0, yn-y0)
    def pt(u, v):
        return (x + (u*np.cos(ang)+v*np.sin(ang))/ratio,
                y + (-u*np.sin(ang)+v*np.cos(ang)))
    tip, tail, left, right = pt(0,size*.52),pt(0,-size*.28),pt(-size*.18,-size*.5),pt(size*.18,-size*.5)
    for pts, color in [([tip,tail,left],'#1E2930'),([tip,right,tail],'white')]:
        ax.add_patch(Polygon(pts,closed=True,transform=ax.transAxes,facecolor=color,
            edgecolor='#1E2930',linewidth=.65,zorder=30,clip_on=False))
    ax.text(*pt(0,size*1.12),'N',transform=ax.transAxes,ha='center',va='center',
        color='black',weight='bold',fontsize=8.5,zorder=31)


def study():
    # Work in a private copy of the small geographic cache; no original file edits.
    cache = OUT / 'source_data/study_area'
    shutil.copytree(s.CACHE, cache, dirs_exist_ok=True)
    s.CACHE=cache; s.OUT=OUT; s.DEST=OUT/'qa/study_area/render'; s.QA=OUT/'qa/study_area'
    s.DEST.mkdir(parents=True,exist_ok=True)
    s.QA.mkdir(parents=True,exist_ok=True)
    s.INK='#26333B'; s.CORE='#AD3B37'; s.north=compass
    def study_label(ax, letter, title):
        label(ax,letter)
        if letter in ('b','c'):
            lon,lat = (2.35,48.876) if letter=='b' else (2.366,48.885)
            x,y=s.PROJ.transform(lon,lat)
            ax.text(x,y,'Paris',ha='center',va='center',color='#B7262C',
                fontsize=8 if letter=='b' else 11,fontweight='bold',zorder=25,
                path_effects=[pe.withStroke(linewidth=2,foreground='white')])
    s.panel=study_label
    with mpl.rc_context({'axes.spines.top':True,'axes.spines.right':True}):
        s.main()


def load_morphology():
    mask=gpd.read_file(r.MORPH/'urban_mask_100m_v1/paris_urban_support_mask_100m_v1.gpkg')
    r.CORE_GEOM=mask.geometry.union_all()
    r.MG=gpd.read_file(r.MORPH/'real_homo_design_v2/urban_cells_reference.gpkg')
    r.MG['x_center']=r.MG.geometry.centroid.x
    r.MG['y_center']=r.MG.geometry.centroid.y
    r.TPL=json.loads((r.MORPH/'real_homo_design_v2/homo_template.json').read_text())


def export(fig, axes, stem, **alignment):
    fig.canvas.draw()
    r.require_matplotlib_panel_alignment(fig,axes=axes,
        panel_ids=list(string.ascii_lowercase[:len(axes)]),
        json_out=OUT/'qa'/f'{stem}.alignment.json',
        overlay_svg=OUT/'qa'/f'{stem}.alignment.svg',strict=True,
        tolerance_pt=1.5,gutter_tolerance_pt=1.5,**alignment)
    fig.savefig(OUT/'png'/f'{stem}.png',dpi=300)
    fig.savefig(OUT/'pdf'/f'{stem}.pdf')
    plt.close(fig)
    print('SAVED',stem,flush=True)


def morphology():
    # Four spatial comparisons keep their original panel order and data.
    fig,axs=plt.subplots(2,3,figsize=(183/25.4,115/25.4))
    fig.subplots_adjust(left=.06,right=.984,top=.935,bottom=.145,wspace=.30,hspace=.54)
    axs=axs.ravel()
    specs=[('REAL','H'),('HOMO','H'),('REAL','FAI'),('HOMO','FAI')]
    for i,(case,field) in enumerate(specs):
        ax=axs[i]
        key=f'H_{case}_m' if field=='H' else f'FAI_{case}_1125'
        height=field=='H'
        artist=r.raster_map(ax,r.MG,r.MG[key],Normalize(0,50 if height else 1.5),
            cmap=HEIGHT if height else FAI,scale=False)
        label(ax,string.ascii_lowercase[i],f'{case} · '+('height' if height else 'directional FAI'))
        cb=r.colourbar(fig,ax,artist,'Building height (m)' if height else 'FAI at 112.5°',extend='max')
        cb.ax.tick_params(labelsize=7); cb.ax.xaxis.label.set_size(7.5)
        cb.outline.set_linewidth(.5)
        if i==0:
            compass(ax,x=.956,y=.80,size=.10)
            xl=ax.get_xlim();yl=ax.get_ylim()
            yy=yl[0]+.12*(yl[1]-yl[0]); xx=xl[0]+.6
            ax.plot([xx,xx+5],[yy,yy],color='#26333B',lw=1.2)
            ax.annotate('5 km',(xx+2.5,yy),xytext=(0,3),textcoords='offset points',ha='center',va='bottom',fontsize=7)
    ax=axs[4];label(ax,'e','Height and coverage')
    ax.hexbin(r.MG.LP_REAL_cond,r.MG.H_REAL_m,gridsize=30,mincnt=1,
        cmap='Greys',norm=LogNorm(),linewidths=0,rasterized=True)
    ax.scatter(r.TPL['LP_star_conditional'],r.TPL['H_star_m'],s=42,marker='D',
        c='#A85535',edgecolors='white',linewidths=.8,label='HOMO template')
    ax.set(xlabel='Urban-conditional plan fraction',ylabel='Building height (m)',
        xlim=(0,max(1.05,r.MG.LP_REAL_cond.max()*1.02)))
    ax.legend(loc='upper right',handletextpad=.4)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(4));ax.yaxis.set_major_locator(mticker.MaxNLocator(4))
    ax=axs[5]; label(ax,'f','Directional FAI distribution')
    values=np.sort(r.MG.lambda_f_1125_conditional.to_numpy())
    ax.plot(values,np.arange(1,len(values)+1)/len(values),color='#207C83',label='REAL',lw=1.3)
    ax.axvline(r.TPL['directional_FAI_star_conditional']['lambda_f_1125'],
        color='#B88242',ls='--',label='HOMO',lw=1.2)
    ax.set(xlabel='Urban-conditional FAI',ylabel='Cumulative fraction',ylim=(0,1.04))
    ax.legend(loc='lower right');ax.xaxis.set_major_locator(mticker.MaxNLocator(4))
    r.source(r.MG,'Fig02_morphology')
    export(fig,list(axs),'Fig02_morphology')
    height_3d()


def height_3d():
    # Genuine x/y/height data, not illustrated buildings. All treatment cells used.
    mg=r.MG
    fig=plt.figure(figsize=(183/25.4,88/25.4))
    gs=fig.add_gridspec(1,2,left=.015,right=.985,top=.94,bottom=.18,wspace=.02)
    axes=[]
    for i,case in enumerate(('REAL','HOMO')):
        ax=fig.add_subplot(gs[0,i],projection='3d',proj_type='ortho');axes.append(ax)
        x=(mg.x_center.to_numpy()-r.ORIGIN[0])/1000
        y=(mg.y_center.to_numpy()-r.ORIGIN[1])/1000
        h=mg[f'H_{case}_m'].to_numpy()
        valid=np.isfinite(h)
        # Missing input heights remain blank, exactly as in the existing 2D map.
        # No zero height is invented and the matched HOMO cells are not dropped.
        (OUT/'qa'/f'height_3d_{case}.json').write_text(json.dumps({
            'input_cells':len(h),'finite_height_prisms':int(valid.sum()),
            'blank_undefined_heights':int((~valid).sum()),
            'rule':'Plot all finite heights; undefined heights remain blank; no imputation.'},indent=2))
        x,y,h=x[valid],y[valid],h[valid]
        colors=HEIGHT(Normalize(0,50,clip=True)(h))
        ax.bar3d(x-.05,y-.05,np.zeros(len(h)),.1,.1,h,
            color=colors,shade=True,linewidth=0,zsort='average',rasterized=True)
        for poly in getattr(r.CORE_GEOM,'geoms',[r.CORE_GEOM]):
            xx,yy=poly.exterior.xy
            ax.plot((np.asarray(xx)-r.ORIGIN[0])/1000,(np.asarray(yy)-r.ORIGIN[1])/1000,
                zs=0,color='#5B6267',lw=.4)
        ax.set(xlim=((mg.x_center.min()-r.ORIGIN[0])/1000-.5,(mg.x_center.max()-r.ORIGIN[0])/1000+.5),
            ylim=((mg.y_center.min()-r.ORIGIN[1])/1000-.5,(mg.y_center.max()-r.ORIGIN[1])/1000+.5),
            zlim=(0,max(mg.H_REAL_m.max(),50)*1.06))
        ax.view_init(elev=40,azim=-65)
        ax.set_box_aspect((2.05,1.13,.60))
        ax.set_axis_off()
        ax.text2D(.06,.98,f'({string.ascii_lowercase[i]})  {case}',transform=ax.transAxes,
            fontsize=9,color='black',va='top')
    ca=fig.add_axes([.31,.145,.38,.02])
    cb=fig.colorbar(mpl.cm.ScalarMappable(norm=Normalize(0,50),cmap=HEIGHT),cax=ca,
        orientation='horizontal',extend='max',ticks=[0,10,20,30,40,50])
    cb.set_label('100 m grid-mean building height (m)',fontsize=8)
    fig.text(.5,.025,'Vertical dimension expanded for visibility · Grid prisms, not individual buildings',
        ha='center',fontsize=7,color='#46505A')
    export(fig,axes,'Fig03_height_3D_alternative')


def load_wrf():
    r.TM=np.load(r.ROOT/'new_study/17_transport_metric_candidates/paired_transport_metrics.npz')
    r.TIMES=pd.to_datetime([str(t).replace('_',' ') for t in r.TM['times']])
    r.POST=r.TIMES>=pd.Timestamp('2022-07-17')
    template=np.load(r.MORPH/'wrf_activation_pilot_v1/injection_templates/paris_d03_real_homo_injection_templates.npz')
    r.ACTIVE=template['PRIMARY_ACTIVATION'].astype(bool)&template['TREATMENT_CORE'].astype(bool)
    r.MODEL=np.load(r.STRICT/'strict_model_inputs.npz')
    wg=np.load(r.ROOT/'new_study/15_network_added_value/phase1_1125/wrf_phase1_export.npz')
    r.WX,r.WY=Transformer.from_crs(4326,2154,always_xy=True).transform(wg['lon'],wg['lat'])
    r.SURFACE=np.load(r.DATA/'气象观测/Paris_WRF_validation/processed/wrf_real_surface_97h.npz')
    assert list(r.SURFACE['times'])==list(r.TM['times'])


def wrf():
    load_wrf()
    fig=plt.figure(figsize=(183/25.4,128/25.4))
    gs=fig.add_gridspec(2,3,left=.061,right=.982,top=.92,bottom=.135,wspace=.29,hspace=.49)
    crs=ccrs.LambertConformal(central_longitude=3,central_latitude=46.5,
        standard_parallels=(44,49),false_easting=700000,false_northing=6600000)
    axes=[]
    lo,hi=-.08,.08
    # A neutral central class prevents tiny responses from tinting the entire map.
    levels=np.array([-.08,-.06,-.04,-.03,-.02,-.01,-.005,.005,.01,.02,.03,.04,.06,.08])
    bound=s.layer('departement',['code_insee'])
    paris=bound.loc[bound.code_insee.astype(str)=='75']
    bounds=(r.ORIGIN[0]-19000,r.ORIGIN[1]-18000,r.ORIGIN[0]+19000,r.ORIGIN[1]+18000)
    water=s.layer('surface_hydrographique',bbox=bounds)
    counts={}
    for i,d in enumerate(r.DIRS):
        ax=fig.add_subplot(gs[i//3,i%3],projection=crs);axes.append(ax)
        sel=r.POST&(r.TM['background_bin_deg']==d)&(r.TM['background_speed_ms']>=1)&(r.TM['background_persistence6h']>=.9)
        field=r.TM['delta_along_10_1000'][sel].mean(axis=0)
        assert np.allclose(field[r.ACTIVE],r.MODEL[f'y_10_1000_{int(d*10):04d}'],atol=1e-7)
        # Contour interpolation is display-only; no smoothing or recomputation.
        artist=ax.contourf(r.WX,r.WY,field,levels=levels,cmap='RdBu_r',
            norm=TwoSlopeNorm(0,lo,hi),extend='both',transform=crs,antialiased=False)
        ax.add_geometries(water.geometry,crs=crs,facecolor='#A2C5CA',edgecolor='none',alpha=.5,zorder=2)
        ax.add_geometries(paris.geometry,crs=crs,facecolor='none',edgecolor='#343E45',linewidth=.8,zorder=5)
        u=r.SURFACE['u10_earth'][sel].mean(axis=0);v=r.SURFACE['v10_earth'][sel].mean(axis=0)
        lon2,lat2,_=Geod(ellps='WGS84').fwd(r.SURFACE['lon'],r.SURFACE['lat'],
            np.degrees(np.arctan2(u,v)),np.hypot(u,v)*100)
        x2,y2=Transformer.from_crs(4326,2154,always_xy=True).transform(lon2,lat2)
        vx=(x2-r.WX)/100;vy=(y2-r.WY)/100
        # Regular thinning applies to arrows only, never the response field.
        idx=np.indices(r.WX.shape)
        keep=(idx[0]%4==0)&(idx[1]%4==0)&(abs(r.WX-r.ORIGIN[0])<17500)&(abs(r.WY-r.ORIGIN[1])<15500)
        q=ax.quiver(r.WX[keep],r.WY[keep],vx[keep],vy[keep],angles='xy',
            scale_units='xy',scale=.0015,width=.0032,headwidth=3.8,headlength=4.5,
            color='#344C56',alpha=.90,zorder=4,transform=crs)
        ax.quiverkey(q,.84,1.034,3,'3 m/s',labelpos='W',coordinates='axes',
            fontproperties={'size':6.5},labelsep=.035)
        p=ax.get_position();ratio=p.width*fig.get_figwidth()/(p.height*fig.get_figheight())
        ax.set_xlim(r.ORIGIN[0]-16000,r.ORIGIN[0]+16000)
        halfheight=16000/ratio
        ax.set_ylim(r.ORIGIN[1]-halfheight,r.ORIGIN[1]+halfheight)
        ax.set_aspect('equal',adjustable='datalim')
        gl=ax.gridlines(crs=ccrs.PlateCarree(),draw_labels=True,linewidth=.25,
            color='#82929B',alpha=.4,linestyle=':',x_inline=False,y_inline=False)
        gl.top_labels=False;gl.right_labels=False;gl.rotate_labels=False
        gl.xlocator=mticker.FixedLocator([2.15,2.35,2.55]);gl.ylocator=mticker.FixedLocator([48.75,48.9])
        gl.xlabel_style={'size':6.5};gl.ylabel_style={'size':6.5};gl.xpadding=2;gl.ypadding=2
        ax.spines['geo'].set_linewidth(.6)
        label(ax,string.ascii_lowercase[i],f'{d:g}° inflow')
        p=ax.get_position();height_mm=fig.get_figheight()*25.4
        ca=fig.add_axes([p.x0,p.y0-7.3/height_mm,p.width,1.8/height_mm])
        cb=fig.colorbar(artist,cax=ca,orientation='horizontal',extend='both',spacing='proportional')
        cb.set_ticks([-.08,-.04,0,.04,.08]);cb.ax.tick_params(labelsize=6.5,pad=1)
        cb.set_label('ΔU (m/s)',fontsize=7,labelpad=1);cb.outline.set_linewidth(.4)
        counts[str(d)]={'hours':int(sel.sum()),'core_cells':int(r.ACTIVE.sum()),
            'min':float(field.min()),'max':float(field.max()),'color_min':lo,'color_max':hi,
            'contour_boundaries':levels.tolist(),'neutral_interval':[-.005,.005]}
    ax=fig.add_subplot(gs[1,1]);axes.append(ax);label(ax,'e','Hourly core response')
    vals=r.TM['delta_along_10_1000'][:,r.ACTIVE];ts=r.TIMES[r.POST]
    low,high=np.quantile(vals[r.POST],[.25,.75],axis=1)
    ax.fill_between(ts,low,high,color=r.TEAL,alpha=.18,label='Spatial IQR',lw=.6)
    ax.plot(ts,np.mean(vals[r.POST],axis=1),c=r.TEAL,label='Spatial mean',lw=1.15)
    r.zero(ax);r.date_axis(ax);ax.set_ylabel('ΔU (m/s)',labelpad=2);ax.set_ylim(-.03,.07)
    ax.set_yticks([-.025,0,.025,.05]);ax.tick_params(axis='x',labelsize=6.5)
    ax.legend(loc='upper right',fontsize=6.5)
    ax=fig.add_subplot(gs[1,2]);axes.append(ax);label(ax,'f','Local response distribution')
    arrays=[r.MODEL[f'y_10_1000_{int(d*10):04d}'] for d in r.DIRS]
    r.points_box(ax,arrays,[f'{d:g}°' for d in r.DIRS],[r.DC[d] for d in r.DIRS],'ΔU (m/s)')
    ax.yaxis.labelpad=2;r.zero(ax);ax.set_xlabel('Inflow direction')
    ax.set_yticks([-.04,0,.04])
    # The font and axis rendering changes do not alter any statistical panel.
    export(fig,axes,'Fig05_transport_response')
    (OUT/'qa/wrf_display.json').write_text(json.dumps(counts,indent=2),encoding='utf-8')


if __name__=='__main__':
    selected=sys.argv[1:] or ['study','morphology','wrf']
    if 'study' in selected: study()
    if {'morphology','wrf'} & set(selected): load_morphology()
    if 'morphology' in selected: morphology()
    if 'wrf' in selected: wrf()

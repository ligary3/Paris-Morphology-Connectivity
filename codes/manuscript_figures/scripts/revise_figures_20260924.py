"""Figure-only revision. Executed after embedded common helpers and data loading.

The notebook builder embeds these functions and all common plotting code.
No manuscript text or original figure is overwritten.
"""

from release_paths import resolve_path as _release_path
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm, LightSource
from matplotlib.patches import Polygon, Patch
from matplotlib import patheffects as pe
from pyproj import Geod

BLUE, RED, TEAL, OCHRE = '#3268A0', '#C44F45', '#278578', '#D48B35'
DC = {45.: '#3268A0', 67.5: '#D78232', 90.: '#7858A1', 112.5: '#278578'}
plt.rcParams.update({'font.size':8, 'axes.titlesize':9, 'axes.titlecolor':'black',
                     'text.color':'black', 'axes.labelcolor':'black'})

def axes_grid(rows=2, cols=3, height=137):
    fig, axs = plt.subplots(rows, cols, figsize=(190/25.4, height/25.4), squeeze=False)
    fig.subplots_adjust(left=.075, right=.978, top=.94, bottom=.10,
                        wspace=.34 if cols==3 else .38, hspace=.54)
    for i,ax in enumerate(axs.flat):
        ax.text(-.105,1.065,f'({string.ascii_lowercase[i]})',transform=ax.transAxes,
                fontsize=10,weight='bold',va='bottom')
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
        ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
    return fig,axs

def title(ax, txt):
    ax.set_title(txt,loc='left',pad=10,fontsize=8.5,color='black')

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

def map_frame(ax, scale=False):
    bounds=np.asarray(CORE_GEOM.bounds).reshape(2,2)
    center=(bounds.mean(axis=0)-ORIGIN)/1000
    span=(bounds[1]-bounds[0])/1000*1.055
    p=ax.get_position(); ratio=p.width*ax.figure.get_figwidth()/(p.height*ax.figure.get_figheight())
    width=max(span[0],span[1]*ratio); height=width/ratio
    ax.set_xlim(center[0]-width/2,center[0]+width/2)
    ax.set_ylim(center[1]-height/2,center[1]+height/2)
    ax.set_aspect('equal',adjustable='datalim'); ax.set_xticks([]);ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(False)
    outline(ax); compass(ax)
    if scale:
        x0,x1=ax.get_xlim();y0,y1=ax.get_ylim();x=x0+.05*(x1-x0);y=y0+.09*(y1-y0)
        ax.plot([x,x+5],[y,y],color='#26333C',lw=1.3)
        ax.annotate('5 km',(x+2.5,y),xytext=(0,-4),textcoords='offset points',ha='center',va='top',fontsize=7)

def colourbar(fig,ax,artist,label,extend='neither'):
    p=ax.get_position(); mm=fig.get_figheight()*25.4
    ca=fig.add_axes([p.x0,p.y0-4.2/mm,p.width,1.65/mm])
    cb=fig.colorbar(artist,cax=ca,orientation='horizontal',extend=extend)
    cb.ax.tick_params(labelsize=7,length=2,pad=1)
    cb.set_label(label,fontsize=7.5,labelpad=1)
    cb.locator=matplotlib.ticker.MaxNLocator(4);cb.update_ticks()
    return cb

def save_new(fig,axes,stem):
    for txt in fig.findobj(match=Text):
        txt.set_text(txt.get_text().replace('NO₂','NO$_2$').replace('O₃','O$_3$'))
    fig.canvas.draw()
    require_matplotlib_panel_alignment(fig,axes=list(np.asarray(axes).flat),
        panel_ids=list(string.ascii_lowercase[:np.asarray(axes).size]),
        json_out=OUT/'qa'/f'{stem}.alignment.json',
        overlay_svg=OUT/'qa'/f'{stem}.alignment.svg',strict=True,
        tolerance_pt=1.5,gutter_tolerance_pt=1.5)
    fig.savefig(OUT/'pdf'/f'{stem}.pdf')
    fig.savefig(OUT/'png'/f'{stem}.png',dpi=300)
    plt.close(fig);print('SAVED',stem,flush=True)

def fig04_morphology():
    fig,axs=axes_grid(height=128)
    config=[('H_REAL_m','REAL building height','Height (m)','YlGnBu',50),
            ('LP_REAL_full','REAL building coverage','Whole-cell plan fraction','YlOrBr',.8),
            ('FAI_REAL_1125','REAL directional FAI','Whole-cell FAI · 112.5°','PuRd',1.5)]
    for ax,(key,ttl,lab,cmap,vmax) in zip(axs[0],config):
        art=raster_map(ax,MG,MG[key],Normalize(0,vmax),cmap,True)
        title(ax,ttl);colourbar(fig,ax,art,lab,'max')
    configs=[('H_REAL_m',TPL['H_star_m'],'Building height (m)','Height distribution'),
             ('LP_REAL_cond',TPL['LP_star_conditional'],'Built-area plan fraction','Coverage distribution'),
             ('lambda_f_1125_conditional',TPL['directional_FAI_star_conditional']['lambda_f_1125'],
              'Built-area FAI · 112.5°','FAI distribution')]
    for ax,(key,ref,xlab,ttl) in zip(axs[1],configs):
        vals=np.sort(MG[key].dropna().to_numpy())
        ax.axhspan(.25,.75,color='#EEF1F4',zorder=0)
        ax.plot(vals,np.arange(1,len(vals)+1)/len(vals),color=BLUE,lw=1.6,label='REAL')
        ax.axvline(ref,color=RED,lw=1.5,ls='--',label=f'HOMO = {ref:.2f}' if key=='H_REAL_m' else f'HOMO = {ref:.3f}')
        if key=='LP_REAL_cond':
            # Author-requested display window only. All 8,229 values contribute
            # to the unchanged ECDF; three values >1 lie outside this viewport.
            # This is NOT a data correction or a renormalized subset ECDF.
            assert len(vals)==len(MG)==8229
            assert np.isfinite(vals).all()
            ax.set_xlim(0,1)
            (OUT/'qa/Fig04e_display_window.json').write_text(json.dumps({
                'field':key,'n_total':len(vals),'xlim':[0,1],
                'n_outside_display':int(((vals<0)|(vals>1)).sum()),
                'max_original':float(vals.max()),'data_corrected':False,
                'ecdf_denominator':len(vals)},indent=2),encoding='utf-8')
        title(ax,ttl);ax.set(xlabel=xlab,ylabel='Cumulative fraction',ylim=(0,1.02))
        ax.legend(loc='lower right',fontsize=7,handlelength=1.4)
    source(MG,'Fig04_morphology_cells')
    save_new(fig,axs,'Fig04_morphology_revised')

def fig05_circuit():
    fig,axs=axes_grid(height=133); a=axs.flat
    # One fixed display multiplier, applied to all scenarios and differences.
    scale=1000.;norm=Normalize(0,np.quantile(CIR[['current_real','current_homo']],.99)*scale)
    for ax,col,ttl in zip(axs[0,:2],['current_real','current_homo'],['REAL connectivity','HOMO connectivity']):
        art=raster_map(ax,CIR,CIR[col]*scale,norm,'YlGnBu',True)
        title(ax,ttl);colourbar(fig,ax,art,'Current (×0.001 model units)','max')
    lim=np.quantile(abs(CIR.delta_current),.99)*scale
    art=raster_map(axs[0,2],CIR,CIR.delta_current*scale,TwoSlopeNorm(0,-lim,lim),'RdBu_r',True)
    title(axs[0,2],'REAL − HOMO');colourbar(fig,axs[0,2],art,'Δcurrent (×0.001 model units)','both')
    ax=axs[1,0];title(ax,'Paired connectivity')
    ax.hexbin(CIR.current_homo*scale,CIR.current_real*scale,gridsize=33,mincnt=1,cmap='YlGnBu',norm=LogNorm(),linewidths=0,rasterized=True)
    mx=max(CIR.current_real.max(),CIR.current_homo.max())*scale*1.03
    ax.plot([0,mx],[0,mx],ls='--',color=SLATE,lw=.8)
    ax.set(xlim=(0,mx),ylim=(0,mx),xlabel='HOMO current (×0.001)',ylabel='REAL current (×0.001)')
    ax.text(.04,.93,f'Spearman ρ = {spearmanr(CIR.current_real,CIR.current_homo).statistic:.3f}',transform=ax.transAxes,fontsize=7.5)
    ax=axs[1,1];title(ax,'High-connectivity overlap');vals=[]
    for q in [.1,.05,.01]:
        n=int(np.ceil(len(CIR)*q));r=set(np.argpartition(CIR.current_real.to_numpy(),-n)[-n:]);h=set(np.argpartition(CIR.current_homo.to_numpy(),-n)[-n:]);vals.append(100*len(r&h)/n)
    ax.bar(range(3),vals,width=.58,color=[BLUE,'#6E98BB','#B0C8DB'])
    for i,v in enumerate(vals):ax.text(i,v+2,f'{v:.1f}%',ha='center',fontsize=8)
    ax.set(xticks=range(3),xticklabels=['Top 10%','Top 5%','Top 1%'],ylabel='REAL/HOMO overlap (%)',ylim=(0,105))
    ax=axs[1,2];title(ax,'Resistance and connectivity');t=CIR[CIR.delta_resistance.abs()>0]
    ax.hexbin(t.delta_resistance,t.delta_current*scale,gridsize=32,mincnt=1,cmap='YlGnBu',norm=LogNorm(),linewidths=0,rasterized=True)
    ax.set(xlabel='Δresistance',ylabel='Δcurrent (×0.001)');zero(ax)
    ax.text(.04,.93,f'Spearman ρ = {spearmanr(t.delta_resistance,t.delta_current).statistic:.3f}',transform=ax.transAxes,fontsize=7.5)
    source(CIR,'Fig05_circuit_cells');save_new(fig,axs,'Fig05_circuit_revised')

def fig06_weather():
    fig,axs=axes_grid(2,2,145)
    b=BG.copy();t=pd.to_datetime(b.time_utc.str.replace('_',' '));ax=axs[0,0]
    title(ax,'Common background wind')
    ax.plot(t,b.inflow_from_deg,c='#AAB2BA',lw=.8)
    for d in DIRS:
        s=(b.circuit_inflow_bin_deg==d)&(b.speed_ms>=1)&(b.persistence6h>=.9)&(t>=pd.Timestamp('2022-07-17'))
        ax.scatter(t[s],b.inflow_from_deg[s],s=14,c=DC[d],label=f'{d:g}°',zorder=3)
    ax.set(ylabel='Wind from (°)',ylim=(0,430),yticks=[0,90,180,270,360]);date_axis(ax)
    ax.legend(ncol=4,loc='upper left',fontsize=6.5,columnspacing=.65,handletextpad=.15)
    twin=ax.twinx();twin.plot(t,b.speed_ms,color='#27333D',lw=1,ls='--',alpha=.85)
    twin.set(ylabel='Speed (m/s)',ylim=(0,10));twin.spines['right'].set_visible(True)
    ax.text(.97,.12,'Dashed: wind speed',ha='right',transform=ax.transAxes,fontsize=7)
    for ax,var,lab,ttl in [(axs[0,1],'temp_c','2 m temperature (°C)','Surface temperature'),
                           (axs[1,0],'wind_speed_ms','10 m wind speed (m/s)','Surface wind speed')]:
        z=SURF.groupby('time_utc')[[f'obs_{var}',f'wrf_{var}']].mean()
        ax.plot(z.index,z[f'obs_{var}'],color='#28353F',lw=1.1,marker='o',markersize=2,markevery=6,label='Observed')
        ax.plot(z.index,z[f'wrf_{var}'],color=BLUE,lw=1.3,label='WRF REAL')
        title(ax,ttl);ax.set_ylabel(lab);date_axis(ax);ax.legend(loc='upper left',fontsize=7,ncol=2)
        if var=='temp_c':ax.set_ylim(10,48)
        else:ax.set_ylim(0,9)
    ax=axs[1,1];title(ax,'Station-level errors')
    names=['Trappes','Toussus','Vélizy','Orly','Le Bourget','Montsouris','CDG']
    # Two aligned mini-axes within one panel avoid putting incompatible units
    # on top of each other. The parent panel preserves the common outer grid.
    p=ax.get_position();ax.set_xticks([]);ax.set_yticks([])
    for sp in ax.spines.values():sp.set_visible(False)
    for j,(vals,col,label,lim) in enumerate([(SM.temp_c_bias,BLUE,'T bias (°C)',(-5,0)),
                                          (SM.wind_speed_ms_rmse,RED,'Wind RMSE (m/s)',(0,2.5))]):
        mini=fig.add_axes([p.x0+j*p.width*.57,p.y0,p.width*.43,p.height*.90])
        mini.scatter(vals,np.arange(7),c=col,s=25,marker='o' if j==0 else 'D')
        mini.set(xlim=lim,ylim=(6.6,-.6),xlabel=label,yticks=range(7))
        mini.set_yticklabels(names if j==0 else ['']*7,fontsize=7)
        mini.tick_params(axis='x',colors=col);mini.xaxis.label.set_color(col)
        mini.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(3))
        mini.grid(axis='y',color='#E8ECEF',lw=.5)
    source(BG,'Fig06_background');source(SURF,'Fig06_surface_pairs');source(SM,'Fig06_station_errors')
    save_new(fig,axs,'Fig06_weather_revised')

def wind_vectors(ax,u,v,step=3,extent=12.5,key=True):
    x=(WX-ORIGIN[0])/1000;y=(WY-ORIGIN[1])/1000
    lon2,lat2,_=Geod(ellps='WGS84').fwd(SURFACE['lon'],SURFACE['lat'],np.degrees(np.arctan2(u,v)),np.hypot(u,v)*100)
    x2,y2=Transformer.from_crs(4326,2154,always_xy=True).transform(lon2,lat2)
    vx=(x2-WX)/100;vy=(y2-WY)/100
    rows,cols=np.indices(x.shape);keep=(rows%step==0)&(cols%step==0)&(abs(x)<extent)&(abs(y-1.3)<extent)
    qu=ax.quiver(x[keep],y[keep],vx[keep],vy[keep],color='#354049',angles='xy',scale_units='xy',scale=2.8,width=.004,headwidth=3.2,zorder=5)
    if key:ax.quiverkey(qu,.64,1.05,3,'3 m/s',labelpos='E',coordinates='axes',fontproperties={'size':7})
    return qu

def wrf_frame(ax,width=25.):
    p=ax.get_position();ratio=p.width*ax.figure.get_figwidth()/(p.height*ax.figure.get_figheight())
    height=width/ratio
    ax.set(xlim=(-width/2,width/2),ylim=(1.3-height/2,1.3+height/2))
    ax.set_aspect('equal',adjustable='datalim');ax.set_xticks([]);ax.set_yticks([])
    for s in ax.spines.values():s.set_visible(True);s.set_color('#ACB5BC');s.set_linewidth(.5)
    outline(ax);compass(ax,x=.95,y=.80,size=.10)
    xl=ax.get_xlim();yl=ax.get_ylim();x=xl[0]+1.;y=yl[0]+1.2
    ax.plot([x,x+5],[y,y],c='#293640',lw=1.2,zorder=20)
    ax.annotate('5 km',(x+2.5,y),xytext=(0,3),textcoords='offset points',ha='center',fontsize=7,zorder=20)

def fig07_transport():
    fig,axs=axes_grid(height=137)
    levels=np.array([-.08,-.06,-.04,-.03,-.02,-.01,-.005,.005,.01,.02,.03,.04,.06,.08])
    cmap=plt.get_cmap('RdBu_r',len(levels)-1);norm=BoundaryNorm(levels,cmap.N)
    for ax,d in zip(list(axs.flat)[:4],DIRS):
        s=POST&(TM['background_bin_deg']==d)&(TM['background_speed_ms']>=1)&(TM['background_persistence6h']>=.9)
        f=TM['delta_along_10_1000'][s].mean(0)
        assert np.allclose(f[ACTIVE],MODEL[f'y_10_1000_{int(d*10):04d}'],atol=1e-7)
        ar=ax.contourf((WX-ORIGIN[0])/1000,(WY-ORIGIN[1])/1000,f,levels=levels,cmap=cmap,norm=norm,extend='both')
        ax.contour((WX-ORIGIN[0])/1000,(WY-ORIGIN[1])/1000,f,levels=[-.04,-.02,.02,.04],colors='#5A6470',linewidths=.35,alpha=.6)
        wind_vectors(ax,SURFACE['u10_earth'][s].mean(0),SURFACE['v10_earth'][s].mean(0))
        wrf_frame(ax);title(ax,f'{d:g}° inflow')
        cb=colourbar(fig,ax,ar,'Δlayer velocity (m/s)','both');cb.set_ticks([-.08,-.04,0,.04,.08])
    vals=TM['delta_along_10_1000'][:,ACTIVE];ts=TIMES[POST];lo,hi=np.quantile(vals[POST],[.25,.75],axis=1)
    ax=axs[1,1];title(ax,'Hourly core response')
    ax.fill_between(ts,lo,hi,color=BLUE,alpha=.18,label='Spatial IQR')
    ax.plot(ts,vals[POST].mean(1),c=BLUE,label='Spatial mean');zero(ax);date_axis(ax)
    ax.set_ylabel('Δlayer velocity (m/s)');ax.set_ylim(-.032,.062);ax.legend(loc='upper right',fontsize=7)
    ax=axs[1,2];title(ax,'Local response distributions')
    arrays=[MODEL[f'y_10_1000_{int(d*10):04d}'] for d in DIRS]
    points_box(ax,arrays,[f'{d:g}°' for d in DIRS],[DC[d] for d in DIRS],'Δlayer velocity (m/s)');zero(ax)
    source(pd.DataFrame({str(d):v for d,v in zip(DIRS,arrays)}),'Fig07_directional_response')
    save_new(fig,axs,'Fig07_transport_revised')

def fig08_added_value():
    fig,axs=axes_grid(2,2,145)
    ax=axs[0,0];title(ax,'Spatial validation blocks')
    blocks=KMeans(n_clusters=5,random_state=0,n_init=20).fit_predict(XY)
    colors=[BLUE,OCHRE,'#7660A4',TEAL,RED];f=np.full(ACTIVE.shape,np.nan);f[ACTIVE]=blocks
    ax.pcolormesh((WX-ORIGIN[0])/1000,(WY-ORIGIN[1])/1000,np.ma.masked_invalid(f),
        cmap=ListedColormap(colors),norm=Normalize(-.5,4.5),shading='nearest',rasterized=True)
    map_frame(ax)
    ax.legend(handles=[Patch(color=c,label=str(i+1)) for i,c in enumerate(colors)],ncol=5,
        title='Validation block · all 85 cells',loc='lower center',bbox_to_anchor=(.5,-.14),fontsize=7,title_fontsize=7,handlelength=.8)
    ax=axs[0,1];title(ax,'Paired prediction errors');q=TEST[TEST.metric=='10_1000']
    # Each repeated spatial split occupies a separate row, avoiding overplotting.
    # Both endpoints retain their original absolute RMSE; no jitter in RMSE.
    assert len(q)==40
    for j,d in enumerate(DIRS):
        z=q[q.direction==d];assert len(z)==10
        rows=j*13+np.arange(len(z))
        ax.axhspan(j*13-.8,j*13+9.8,color=DC[d],alpha=.055,lw=0,zorder=0)
        ax.hlines(rows,z.rmse_circuit,z.rmse_baseline,color=DC[d],lw=.85,zorder=2)
        ax.scatter(z.rmse_baseline,rows,s=6,facecolors='white',edgecolors=DC[d],lw=.5,zorder=3)
        ax.scatter(z.rmse_circuit,rows,s=6,c=DC[d],marker='D',linewidths=0,zorder=4)
    ax.set(yticks=np.arange(4)*13+4.5,yticklabels=[f'{d:g}°' for d in DIRS],
        xlabel='Held-out RMSE (m/s)',xlim=(.012,.0245),ylim=(50,-7),
        xticks=[.012,.015,.018,.021,.024])
    ax.tick_params(axis='y',length=0,pad=5)
    ax.spines['left'].set_visible(False)
    ax.legend(handles=[Line2D([],[],marker='o',markersize=3.5,mfc='white',mec='#444444',ls='',label='Baseline'),
        Line2D([],[],marker='D',markersize=3.5,color='#444444',ls='',label='+ Circuit')],
        ncol=2,loc='upper left',fontsize=7,borderaxespad=.1,handletextpad=.4,columnspacing=1.2)
    ax=axs[1,0];title(ax,'Network gain by direction')
    points_box(ax,[-100*q[q.direction==d].relative_rmse_change.to_numpy() for d in DIRS],
        [f'{d:g}°' for d in DIRS],[DC[d] for d in DIRS],'RMSE reduction (%)');zero(ax)
    ax=axs[1,1];title(ax,'Network–response association')
    x=MODEL['network_1125'].ravel()*1000;y=MODEL['y_10_1000_1125']
    ax.scatter(x,y,s=21,c=TEAL,alpha=.75,edgecolors='white',lw=.35)
    xx=np.linspace(x.min(),x.max(),100);ax.plot(xx,np.polyval(np.polyfit(x,y,1),xx),c=RED,lw=1.2,label='Descriptive linear fit')
    ax.text(.04,.94,f'Spearman ρ = {spearmanr(x,y).statistic:.3f}',transform=ax.transAxes,fontsize=8)
    ax.set(xlabel='Δcurrent (×0.001; 112.5°)',ylabel='Δlayer velocity (m/s)',ylim=(-.05,.07));zero(ax);ax.legend(loc='lower left',fontsize=7)
    source(pd.DataFrame({'x':XY[:,0],'y':XY[:,1],'block':blocks+1}),'Fig08_blocks_all85')
    source(q,'Fig08_cv');save_new(fig,axs,'Fig08_added_value_revised')

def fig09_lcz():
    fig,axs=axes_grid(2,2,144);ax=axs[0,0];title(ax,'Local climate zones')
    cs=LCZ[LCZ.analysis_core.astype(bool)].copy();cats=sorted(cs.lcz_int.unique())
    palette=['#9F3E32','#CF5D48','#E4967B','#A77593','#C9A0B5','#E0BCCB','#BAAF95','#CFB486','#ACA9A3','#645956','#3D7656','#91B184','#C1D09F','#DBDFB7','#B8C3C9','#C0A479','#75AFC9']
    raster_map(ax,cs,[cats.index(c) for c in cs.lcz_int],Normalize(-.5,len(cats)-.5),ListedColormap([palette[int(c)-1] for c in cats]))
    handles=[Patch(color=palette[int(c)-1],label=str(c) if c<=10 else chr(65+int(c)-11)) for c in cats]
    ax.legend(handles=handles,ncol=5,loc='lower center',bbox_to_anchor=(.5,-.22),fontsize=7,handlelength=.9,columnspacing=1)
    ax=axs[0,1];title(ax,'Class-based versus detailed FAI')
    ca=read_csv(DATA/'LCZ/Paris/Cerema_2022/circuit_benchmark_v1/1125/cells.csv')
    # Match the original input-audit population: the same 8,229 treatment cells.
    # Untreated zero-FAI cells must not inflate the class-compression correlation.
    ca=ca[ca.grid_id.isin(MG.grid_id)]
    ax.hexbin(ca.fai_real,ca.fai_lcz,gridsize=32,mincnt=1,cmap='YlGnBu',norm=LogNorm(),linewidths=0,rasterized=True)
    ax.text(.04,.94,f'Spearman ρ = {spearmanr(ca.fai_real,ca.fai_lcz).statistic:.3f}',transform=ax.transAxes,fontsize=8)
    ax.set(xlabel='Detailed 3D FAI',ylabel='LCZ FAI · 112.5°',ylim=(-.04,max(1.05,float(ca.fai_lcz.max())*1.20)))
    ax=axs[1,0];title(ax,'All 40 matched comparisons')
    points_box(ax,[-100*LTEST.relative_change_lcz.to_numpy(),-100*LTEST.relative_change_3d.to_numpy()],
        ['LCZ','Detailed 3D'],[OCHRE,BLUE],'RMSE reduction (%)');zero(ax)
    ax=axs[1,1];title(ax,'Directional network gains')
    for offset,col,key,label in [(-.20,OCHRE,'relative_change_lcz','LCZ'),(.20,BLUE,'relative_change_3d','Detailed 3D')]:
        arrays=[-100*LTEST[LTEST.direction==d][key].to_numpy() for d in DIRS]
        bp=ax.boxplot(arrays,positions=np.arange(4)+offset,widths=.30,patch_artist=True,showfliers=False,
            medianprops={'color':col,'lw':1.5},whiskerprops={'color':col,'lw':.8},capprops={'color':col,'lw':.8})
        for box in bp['boxes']:box.set_facecolor(col);box.set_alpha(.20);box.set_edgecolor(col)
        for j,arr in enumerate(arrays):ax.scatter(j+offset+np.linspace(-.065,.065,len(arr)),arr,s=11,c=col,alpha=.85,edgecolors='white',lw=.25,zorder=4)
    ax.set(xticks=range(4),xticklabels=[f'{d:g}°' for d in DIRS],ylabel='RMSE reduction (%)',ylim=(-2,15));zero(ax)
    ax.legend(handles=[Patch(color=OCHRE,label='LCZ'),Patch(color=BLUE,label='Detailed 3D')],ncol=2,fontsize=7,loc='upper left')
    source(ca,'Fig09_fai');source(LTEST,'Fig09_lcz_cv');save_new(fig,axs,'Fig09_lcz_revised')

def demean(z,key):
    return z[key]-z.groupby('station_code')[key].transform('mean')-z.groupby('time_utc')[key].transform('mean')+z[key].mean()

def fig10_air_quality():
    fig,axs=axes_grid(height=139);base=DATA/'空气质量/Paris_2022_event/analysis'
    pairs=read_csv(base/'D075_background_observation_wrf_pairs.csv');st=read_csv(base/'D075_station_wrf_grid_match.csv')
    res=json.loads((base/'air_quality_results.json').read_text())['contemporaneous_double_demeaned_results']
    ax=axs[0,0];title(ax,'Urban-background stations');map_frame(ax,True)
    xx,yy=Transformer.from_crs(4326,2154,always_xy=True).transform(st.longitude,st.latitude)
    ozone=set(pairs.loc[pairs.O3.notna(),'station_code'])
    offsets={'PA13':((-9,-20),'center'),'PA18':((-8,10),'right'),'PA15L':((-5,-10),'right'),
             'PA12':((13,21),'left'),'PA07':((3,-10),'left'),'PA01H':((-1,6),'center')}
    for x,y,code in zip(xx,yy,st.station_code):
        x=(x-ORIGIN[0])/1000;y=(y-ORIGIN[1])/1000
        ax.scatter(x,y,marker='D' if code in ozone else 'o',s=32,c=TEAL,edgecolors='white',lw=.5)
        off,ha=offsets[code]
        leader={'arrowstyle':'-','color':'#78858C','lw':.55,'shrinkA':2,'shrinkB':4} if code in ['PA12','PA13'] else None
        ax.annotate(code,(x,y),xytext=off,textcoords='offset points',fontsize=7,ha=ha,arrowprops=leader)
    ax.legend(handles=[Line2D([],[],ls='',marker='D',color=TEAL,label='NO$_2$ / O$_3$'),Line2D([],[],ls='',marker='o',color=TEAL,label='NO$_2$ only')],loc='lower center',bbox_to_anchor=(.5,-.22),fontsize=7.3,ncol=2,handletextpad=.15,columnspacing=.5)
    ax=axs[0,1];title(ax,'Observed ozone')
    for code,col,ls in zip(sorted(ozone),[BLUE,OCHRE,TEAL],['-','--',':']):
        z=pairs[pairs.station_code==code];ax.plot(pd.to_datetime(z.time_utc),z.O3,label=code,c=col,ls=ls)
    ax.set(ylabel='O$_3$ (µg/m³)',ylim=(0,200));date_axis(ax);ax.legend(ncol=3,loc='upper left',fontsize=6.5,columnspacing=.6,handlelength=1)
    formal=pairs[pairs.background_bin_deg.isin(DIRS)]
    configs=[('NO2','NO$_2$','µg/m³',BLUE),('O3','O$_3$','µg/m³',TEAL),('Ox_umol_m3','Ox','µmol/m³',OCHRE)]
    for ax,(key,label,unit,col) in zip([axs[0,2],axs[1,0],axs[1,1]],configs):
        z=formal.dropna(subset=[key,'delta_transport']).copy();x=demean(z,'delta_transport');y=demean(z,key)
        ax.scatter(x,y,s=10,c=col,alpha=.42,lw=0,rasterized=True,label=f'Station–hours (n = {len(z)})')
        xx=np.linspace(x.min(),x.max(),100);ax.plot(xx,np.polyval(np.polyfit(x,y,1),xx),c=col,lw=1.6,label='Descriptive linear fit')
        zero(ax);title(ax,f'{label} association')
        ax.set(xlabel='Δvelocity residual (m/s)',ylabel=f'{label} residual ({unit})')
        span=y.max()-y.min();ax.set_ylim(y.min()-.05*span,y.max()+.25*span)
        ax.legend(loc='upper right',fontsize=6.3,handlelength=1.3,markerscale=1.3)
        source(pd.DataFrame({'station':z.station_code,'time':z.time_utc,'x':x,'y':y}),f'Fig10_{key}_residuals')
    ax=axs[1,2];title(ax,'Association and uncertainty')
    for i,r in enumerate(res):
        v=r['spearman_rho_double_demeaned'];lo=r['block_bootstrap_ci_low'];hi=r['block_bootstrap_ci_high'];col=configs[i][3]
        ax.axhspan(i-.35,i+.35,color='#F3F5F6',zorder=0)
        ax.errorbar(v,i,xerr=[[v-lo],[hi-v]],fmt='o',c=col,capsize=3,ms=5,lw=1.6)
        ax.text(.56,i,f'{v:.3f}',ha='right',va='center',fontsize=7,color=col)
    ax.set(yticks=range(3),yticklabels=['NO$_2$','O$_3$','Ox'],ylim=(2.7,-.7),xlim=(-.8,.6),xlabel='Spearman ρ (95% block CI)');zero(ax,'x')
    source(pd.DataFrame(res),'Fig10_block_intervals');save_new(fig,axs,'Fig10_air_quality_revised')

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
    if labels:ax.set_xticks(xt,xl,fontsize=6);ax.set_yticks(yt,yl,fontsize=6);ax.tick_params(length=2,pad=1)

def fig02_study():
    import rasterio,pyogrio
    from shapely.geometry import box
    cache=ROOT/'manuscript_figures/source_data/study_area'
    paris=gpd.read_file(cache/'D075.geojson').to_crs(2154)
    circ=gpd.read_file(cache/'circuit_1125_domain.geojson').to_crs(2154)
    dom=np.load(cache/'wrf_domains.npz');stations=pd.read_csv(cache/'meteorological_stations.csv')
    raw=next((DATA/'建筑形态/Paris/BDTOPO_3-0_TOUSTHEMES_GPKG_LAMB93_D075_2022-06-15').rglob('*.gpkg'))
    bounds=paris.total_bounds; bb=box(*bounds).buffer(1500).bounds
    def layer(name):
        columns={'zone_de_vegetation':['nature'],'zone_d_activite_ou_d_interet':['nature_detaillee']}.get(name,[])
        return pyogrio.read_dataframe(raw,layer=name,bbox=bb,columns=columns).to_crs(2154)
    buildings=layer('batiment');water=layer('surface_hydrographique');veg=layer('zone_de_vegetation')
    veg=veg[veg.nature.fillna('')!='Haie'];parks=layer('zone_d_activite_ou_d_interet');parks=parks[parks.nature_detaillee.fillna('')=='Parc']
    fig=plt.figure(figsize=(190/25.4,120/25.4));gs=fig.add_gridspec(2,2,width_ratios=[1,1.87],left=.065,right=.985,bottom=.17,top=.95,wspace=.16,hspace=.27)
    a=fig.add_subplot(gs[0,0]);b=fig.add_subplot(gs[1,0]);c=fig.add_subplot(gs[:,1])
    def fit(ax,bounds):
        x0,y0,x1,y1=bounds;p=ax.get_position();ratio=p.width*fig.get_figwidth()/(p.height*fig.get_figheight());w=max(x1-x0,(y1-y0)*ratio);h=w/ratio
        ax.set(xlim=((x0+x1-w)/2,(x0+x1+w)/2),ylim=((y0+y1-h)/2,(y0+y1+h)/2));ax.set_aspect('equal',adjustable='datalim')
        for sp in ax.spines.values():sp.set_visible(True);sp.set_color('#AFBBC1');sp.set_linewidth(.5)
    ne=_release_path('C:/Users/12169/.local/share/cartopy/shapefiles/natural_earth');eu=box(-25,32,25,65)
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
    save_new(fig,[a,b,c],'Fig02_study_area_revised')

def fig_wrf_states():
    path=OUT/'source_data/wrf_paired_snapshot.npz'
    if not path.exists():
        print('SKIPPED absolute WRF state maps: snapshot export not available',flush=True);return
    z=np.load(path);fig,axs=axes_grid(height=140)
    fields=[('layer_speed','Layer-mean wind','Wind speed (m/s)','YlGnBu'),
            ('t2','2 m temperature','Temperature (°C)','YlOrRd'),('pblh','PBL height','PBLH (m)','YlGn')]
    x=(WX-ORIGIN[0])/1000;y=(WY-ORIGIN[1])/1000
    mask=(abs(x)<13)&(abs(y-1.3)<12)
    for j,(key,ttl,label,cmap) in enumerate(fields):
        values=np.r_[z[f'REAL_{key}'][mask],z[f'HOMO_{key}'][mask]]
        levels=np.linspace(np.floor(values.min()*10)/10,np.ceil(values.max()*10)/10,12)
        for i,case in enumerate(['REAL','HOMO']):
            ax=axs[i,j];ar=ax.contourf(x,y,z[f'{case}_{key}'],levels=levels,cmap=cmap,extend='both')
            if key=='layer_speed':
                qu=wind_vectors(ax,z[f'{case}_u_layer'],z[f'{case}_v_layer'],key=False)
                qk=ax.quiverkey(qu,.72,.055,3,'3 m/s',labelpos='E',coordinates='axes',fontproperties={'size':7})
                qk.text.set_bbox(dict(facecolor='white',edgecolor='none',alpha=.85,pad=1))
            wrf_frame(ax);title(ax,f'{case} · '+ttl)
            cb=colourbar(fig,ax,ar,label,'both');cb.minorticks_off()
            cb.set_ticks({'layer_speed':[3.6,4.0,4.4,4.8],'t2':[24,27,30,33],'pblh':[700,800,900,1000]}[key])
    save_new(fig,axs,'Fig_new_WRF_REAL_HOMO_states')

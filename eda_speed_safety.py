#!/usr/bin/env python
import os, ast, numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = r'C:/Users/chois/Gitsrcs/Swingdata'
OUT_DIR = os.path.join(DATA_DIR, 'eda_output')
os.makedirs(OUT_DIR, exist_ok=True)
ROUTES = os.path.join(DATA_DIR, '2023_05_Swing_Routes.csv')
SCOOTER = os.path.join(DATA_DIR, '2023_Swing_Scooter.csv')

plt.rcParams.update({'figure.dpi': 300, 'savefig.dpi': 300, 'font.size': 10,
    'axes.titlesize': 12, 'axes.labelsize': 10, 'xtick.labelsize': 9,
    'ytick.labelsize': 9, 'legend.fontsize': 9, 'figure.facecolor': 'white',
    'axes.facecolor': 'white', 'axes.grid': True, 'grid.alpha': 0.3,
    'axes.spines.top': False, 'axes.spines.right': False})
C = ['#648FFF', '#DC267F', '#FE6100', '#FFB000', '#785EF0', '#009E73']
pcts = [5, 25, 50, 75, 95, 99]

print('=' * 72)
print('  E-SCOOTER SAFETY EDA - SWING DATA (MAY 2023)')
print('=' * 72)

print('\n' + '=' * 72)
print('  PART 1: ROUTE-LEVEL SPEED ANALYSIS (100K SAMPLE)')
print('=' * 72)

rcols = ['model','mode','travel_time','distance','moved_distance',
         'points','avg_speed','max_speed','avg_point_gap','start_x','start_y']
df = pd.read_csv(ROUTES, usecols=rcols, nrows=100_000)
print(f'\nLoaded {len(df):,} route records.')
print(f'\nDtypes:\n{df.dtypes}')
print(f'\nMissing:\n{df.isnull().sum()}')

for c in ['travel_time','distance','moved_distance','points',
          'avg_speed','max_speed','avg_point_gap','start_x','start_y']:
    df[c] = pd.to_numeric(df[c], errors='coerce')

vm = (df['avg_speed'] > 0) & (df['max_speed'] > 0)
dv = df[vm].copy()
print(f'\nValid: {len(dv):,} / {len(df):,} ({100*len(dv)/len(df):.1f}%)')
nt = len(dv)

print('\n' + '-' * 60)
print('  1a. SPEED DISTRIBUTION SUMMARY')
print('-' * 60)
for col, label in [('avg_speed','Average Speed (km/h)'),('max_speed','Maximum Speed (km/h)')]:
    v = dv[col].dropna()
    print(f'\n  {label}:')
    print(f'    Count : {len(v):,}')
    print(f'    Mean  : {v.mean():.2f}')
    print(f'    Median: {v.median():.2f}')
    print(f'    Std   : {v.std():.2f}')
    print(f'    Min   : {v.min():.2f}')
    print(f'    Max   : {v.max():.2f}')
    for p in pcts:
        print(f'    P{p:02d}   : {np.percentile(v, p):.2f}')

print('\n' + '-' * 60)
print('  1b. SPEED LIMIT EXCEEDANCE')
print('-' * 60)
print('  By max_speed:')
for lim in [15, 20, 25]:
    ne = (dv['max_speed'] > lim).sum()
    print(f'    > {lim} km/h: {ne:,} / {nt:,} ({100*ne/nt:.1f}%)')
print('  By avg_speed:')
for lim in [15, 20, 25]:
    ne = (dv['avg_speed'] > lim).sum()
    print(f'    > {lim} km/h: {ne:,} / {nt:,} ({100*ne/nt:.1f}%)')

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, col, label, color in zip(axes, ['avg_speed','max_speed'],
    ['Average Speed (km/h)','Maximum Speed (km/h)'], [C[0], C[1]]):
    v = dv[col].dropna(); vc = v[v <= 50]
    ax.hist(vc, bins=100, color=color, alpha=0.7, edgecolor='white', linewidth=0.3)
    ax.axvline(v.median(), color='black', ls='--', lw=1.2, label=f'Med={v.median():.1f}')
    for lm, ls2 in [(15,':'),(20,'-.'),(25,'--')]:
        ax.axvline(lm, color='red', ls=ls2, lw=1, alpha=0.7, label=f'{lm} km/h')
    ax.set_xlabel(label); ax.set_ylabel('Trips')
    ax.set_title(f'Distribution of {label}'); ax.legend(fontsize=8)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR,'fig01_speed_distributions.png'), dpi=300, bbox_inches='tight')
plt.close(fig); print('  [Saved] fig01_speed_distributions.png')

print('\n' + '-' * 60)
print('  1c. SPEED BY VEHICLE MODEL')
print('-' * 60)
mc = dv['model'].value_counts()
print('\n  Model distribution:')
for m, cnt in mc.items(): print(f'    {m}: {cnt:,} ({100*cnt/nt:.1f}%)')
ms = dv.groupby('model').agg(
    n=('avg_speed','count'), avg_mean=('avg_speed','mean'),
    avg_med=('avg_speed','median'), avg_std=('avg_speed','std'),
    max_mean=('max_speed','mean'), max_med=('max_speed','median'),
    max_p95=('max_speed', lambda x: np.percentile(x,95)),
    max_p99=('max_speed', lambda x: np.percentile(x,99))).round(2)
print('\n  Speed stats by model:'); print(ms.to_string())

tm = mc.head(6).index.tolist(); dt = dv[dv['model'].isin(tm)]
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, col, label in zip(axes, ['avg_speed','max_speed'],
    ['Average Speed (km/h)','Maximum Speed (km/h)']):
    dbm = [dt.loc[dt['model']==m, col].dropna().values for m in tm]
    bp = ax.boxplot(dbm, labels=tm, patch_artist=True, showfliers=False,
                    widths=0.6, medianprops=dict(color='black', lw=1.5))
    for ptch, clr in zip(bp['boxes'], C[:len(tm)]):
        ptch.set_facecolor(clr); ptch.set_alpha(0.7)
    for lm in [15,20,25]: ax.axhline(lm, color='red', ls=':', lw=0.8, alpha=0.5)
    ax.set_xlabel('Model'); ax.set_ylabel(label); ax.set_title(f'{label} by Model')
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR,'fig02_speed_by_model.png'), dpi=300, bbox_inches='tight')
plt.close(fig); print('  [Saved] fig02_speed_by_model.png')

print('\n' + '-' * 60)
print('  1d. SPEED BY MODE')
print('-' * 60)
dv['mode_c'] = dv['mode'].fillna('none')
moc = dv['mode_c'].value_counts()
print('\n  Mode distribution:')
for m, cnt in moc.items(): print(f'    {m}: {cnt:,} ({100*cnt/nt:.1f}%)')
mos = dv.groupby('mode_c').agg(
    n=('avg_speed','count'), avg_mean=('avg_speed','mean'),
    avg_med=('avg_speed','median'), max_mean=('max_speed','mean'),
    max_med=('max_speed','median'),
    max_p95=('max_speed', lambda x: np.percentile(x,95)),
    exceed25=('max_speed', lambda x: (x>25).mean()*100)).round(2)
print('\n  Speed stats by mode:'); print(mos.to_string())

ml = moc.index.tolist()
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, col, label in zip(axes, ['avg_speed','max_speed'],
    ['Average Speed (km/h)','Maximum Speed (km/h)']):
    dbmo = [dv.loc[dv['mode_c']==m, col].dropna().values for m in ml]
    bp = ax.boxplot(dbmo, labels=ml, patch_artist=True, showfliers=False,
                    widths=0.5, medianprops=dict(color='black', lw=1.5))
    for ptch, clr in zip(bp['boxes'], C[:len(ml)]):
        ptch.set_facecolor(clr); ptch.set_alpha(0.7)
    for lm in [15,20,25]: ax.axhline(lm, color='red', ls=':', lw=0.8, alpha=0.5)
    ax.set_xlabel('Mode'); ax.set_ylabel(label); ax.set_title(f'{label} by Mode')
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR,'fig03_speed_by_mode.png'), dpi=300, bbox_inches='tight')
plt.close(fig); print('  [Saved] fig03_speed_by_mode.png')

print('\n' + '-' * 60)
print('  1e. SPEED vs DISTANCE / TRAVEL TIME')
print('-' * 60)
cd = dv[['avg_speed','distance']].corr().iloc[0,1]
cm = dv[['avg_speed','moved_distance']].corr().iloc[0,1]
ct = dv[['avg_speed','travel_time']].corr().iloc[0,1]
print(f'  corr(avg_speed, distance)      : {cd:.4f}')
print(f'  corr(avg_speed, moved_distance): {cm:.4f}')
print(f'  corr(avg_speed, travel_time)   : {ct:.4f}')

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
np.random.seed(42)
sidx = np.random.choice(len(dv), size=min(10000,len(dv)), replace=False)
ds = dv.iloc[sidx]
ax=axes[0]
ax.scatter(ds['distance']/1000, ds['avg_speed'], s=3, alpha=0.15, color=C[0], rasterized=True)
ax.set_xlabel('Distance (km)'); ax.set_ylabel('Avg Speed (km/h)')
ax.set_title(f'Speed vs Dist (r={cd:.3f})')
ax.set_xlim(0, ds['distance'].quantile(0.99)/1000); ax.set_ylim(0,40)
ax=axes[1]
ax.scatter(ds['travel_time']/60, ds['avg_speed'], s=3, alpha=0.15, color=C[1], rasterized=True)
ax.set_xlabel('Travel Time (min)'); ax.set_ylabel('Avg Speed (km/h)')
ax.set_title(f'Speed vs Time (r={ct:.3f})')
ax.set_xlim(0, ds['travel_time'].quantile(0.99)/60); ax.set_ylim(0,40)
ax=axes[2]
ratio = ds['moved_distance'] / ds['distance'].replace(0, np.nan)
vr = ratio.dropna()
ax.scatter(vr, ds.loc[vr.index,'avg_speed'], s=3, alpha=0.15, color=C[2], rasterized=True)
ax.set_xlabel('Moved/GPS Dist Ratio'); ax.set_ylabel('Avg Speed (km/h)')
ax.set_title('Speed vs Dist Ratio'); ax.set_xlim(0,5); ax.set_ylim(0,40)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR,'fig04_speed_vs_trip_features.png'), dpi=300, bbox_inches='tight')
plt.close(fig); print('  [Saved] fig04_speed_vs_trip_features.png')

print('\n' + '-' * 60)
print('  1f. DATA QUALITY')
print('-' * 60)
for col, label in [('points','Points per trip'),('avg_point_gap','Avg point gap (sec)')]:
    v = dv[col].dropna()
    print(f'\n  {label}: Mean={v.mean():.2f}, Median={v.median():.2f}, Std={v.std():.2f}')
    for p in pcts: print(f'    P{p:02d}={np.percentile(v,p):.2f}')

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
ax=axes[0]; pts=dv['points'].dropna(); ptc=pts[pts<=pts.quantile(0.99)]
ax.hist(ptc, bins=80, color=C[0], alpha=0.7, edgecolor='white', lw=0.3)
ax.set_xlabel('GPS Points per Trip'); ax.set_ylabel('Count')
ax.set_title('GPS Points per Trip')
ax.axvline(pts.median(), color='black', ls='--', lw=1.2, label=f'Med={pts.median():.0f}')
ax.legend()
ax=axes[1]; gap=dv['avg_point_gap'].dropna(); gpc=gap[gap<=gap.quantile(0.99)]
ax.hist(gpc, bins=80, color=C[3], alpha=0.7, edgecolor='white', lw=0.3)
ax.set_xlabel('Avg Point Gap (sec)'); ax.set_ylabel('Count')
ax.set_title('Avg GPS Point Gap')
ax.axvline(gap.median(), color='black', ls='--', lw=1.2, label=f'Med={gap.median():.1f}s')
ax.axvline(10, color='red', ls=':', lw=1, label='10s expected'); ax.legend()
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR,'fig05_data_quality.png'), dpi=300, bbox_inches='tight')
plt.close(fig); print('  [Saved] fig05_data_quality.png')

print('\n' + '-' * 60)
print('  1g. GEOGRAPHIC SPREAD')
print('-' * 60)
lat=dv['start_x'].dropna(); lon=dv['start_y'].dropna()
print(f'  Lat: [{lat.min():.4f}, {lat.max():.4f}]')
print(f'  Lon: [{lon.min():.4f}, {lon.max():.4f}]')
dv['lat_r']=(dv['start_x']*10).round()/10
dv['lon_r']=(dv['start_y']*10).round()/10
gcl=dv.groupby(['lat_r','lon_r']).size().reset_index(name='cnt')
gcl=gcl.sort_values('cnt',ascending=False)
print(f'\n  Grid clusters: {len(gcl)}')
cmap_city = {(37.6,127.0):'Seoul',(37.5,127.0):'Seoul',(37.4,127.0):'Seoul/Gangnam',
    (37.5,126.9):'Seoul',(37.4,126.9):'Seoul',(35.2,129.1):'Busan',
    (35.1,129.0):'Busan',(36.3,127.4):'Daejeon',(36.4,127.4):'Daejeon',
    (35.9,128.6):'Daegu',(33.5,126.5):'Jeju',(33.5,126.6):'Jeju',
    (37.3,127.0):'Suwon',(37.4,127.1):'Seongnam',
    (37.1,127.1):'Pyeongtaek',(37.1,127.0):'Pyeongtaek',
    (35.5,129.3):'Ulsan',(36.8,127.0):'Cheonan'}
print('\n  Top 20 clusters:')
for _, r in gcl.head(20).iterrows():
    k=(r['lat_r'],r['lon_r']); city=cmap_city.get(k,'Unknown')
    print(f"    ({r['lat_r']:.1f},{r['lon_r']:.1f}) => {city}: {r['cnt']:,}")

fig, ax = plt.subplots(figsize=(8, 10))
sg = dv.sample(min(20000,len(dv)), random_state=42)
sc = ax.scatter(sg['start_y'], sg['start_x'], s=1, alpha=0.1,
    c=sg['avg_speed'], cmap='YlOrRd', vmin=0, vmax=30, rasterized=True)
plt.colorbar(sc, ax=ax, shrink=0.6, label='Avg Speed (km/h)')
ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude')
ax.set_title('Geographic Distribution of E-Scooter Trips')
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR,'fig06_geographic_spread.png'), dpi=300, bbox_inches='tight')
plt.close(fig); print('  [Saved] fig06_geographic_spread.png')

print('\n\n' + '=' * 72)
print('  PART 2: INSTANTANEOUS SPEED PROFILES (1K)')
print('=' * 72)

dfs = pd.read_csv(ROUTES,
    usecols=['speeds','avg_speed','max_speed',
             'model','mode','travel_time'],
    nrows=1000)
print(f'\nLoaded {len(dfs):,} rows.')

def parse_speeds(s2):
    if pd.isna(s2): return []
    try:
        parsed = ast.literal_eval(s2)
        if isinstance(parsed, list) and len(parsed) > 0:
            if isinstance(parsed[0], list): return parsed[0]
            return parsed
        return []
    except: return []

dfs['slist'] = dfs['speeds'].apply(parse_speeds)
dfs['npts'] = dfs['slist'].apply(len)
dsp = dfs[dfs['npts'] >= 5].copy()
print(f'Trips >=5 pts: {len(dsp):,}/{len(dfs):,}')

all_sp = []
for sl in dsp['slist']: all_sp.extend(sl)
all_sp = np.array(all_sp, dtype=float)
print(f'Total points: {len(all_sp):,}')

print('\n' + '-' * 60)
print('  2a. INSTANTANEOUS SPEED DISTRIBUTION')
print('-' * 60)
print(f'  Mean  : {all_sp.mean():.2f} km/h')
print(f'  Median: {np.median(all_sp):.2f} km/h')
print(f'  Std   : {all_sp.std():.2f} km/h')
print(f'  Min   : {all_sp.min():.2f} km/h')
print(f'  Max   : {all_sp.max():.2f} km/h')
for p in pcts:
    print(f'  P{p:02d}: {np.percentile(all_sp,p):.2f}')
print('\n  Exceeding thresholds:')
for lim in [15, 20, 25, 30]:
    frac = (all_sp > lim).mean() * 100
    print(f'    >{lim} km/h: {frac:.1f}%')

fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(all_sp,
    bins=np.arange(0, all_sp.max()+2, 1),
    color=C[0], alpha=0.7,
    edgecolor='white', lw=0.3)
ax.axvline(np.median(all_sp),
    color='black', ls='--', lw=1.2,
    label=f'Med={np.median(all_sp):.1f}')
for lm in [15, 20, 25]:
    ax.axvline(lm, color='red',
        ls=':', lw=1, alpha=0.7,
        label=f'{lm} km/h')
ax.set_xlabel('Instantaneous Speed (km/h)')
ax.set_ylabel('Count')
ax.set_title('Instantaneous Speeds')
ax.legend()
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR,
    'fig07_instantaneous_speed_dist.png'),
    dpi=300, bbox_inches='tight')
plt.close(fig)
print('  [Saved] fig07_instantaneous_speed_dist.png')

print('\n' + '-' * 60)
print('  2b. SPEED PROFILE PATTERNS')
print('-' * 60)
n_bins = 20
prof_mat = []
for sl in dsp['slist']:
    arr = np.array(sl, dtype=float)
    if len(arr) < 5: continue
    xo = np.linspace(0, 1, len(arr))
    xn = np.linspace(0, 1, n_bins)
    prof_mat.append(np.interp(xn, xo, arr))
prof_mat = np.array(prof_mat)
mn_prof = prof_mat.mean(axis=0)
sd_prof = prof_mat.std(axis=0)
p25p = np.percentile(prof_mat, 25, axis=0)
p75p = np.percentile(prof_mat, 75, axis=0)
pos = np.linspace(0, 100, n_bins)
print(f'  Trips: {len(prof_mat):,}')
print('\n  Avg speed profile:')
for i in range(0, n_bins, 4):
    print(f'    {pos[i]:5.1f}%: '+
          f'mean={mn_prof[i]:.1f}, '+
          f'std={sd_prof[i]:.1f}')
fq = mn_prof[:n_bins//4].mean()
mh = mn_prof[n_bins//4:3*n_bins//4].mean()
lq = mn_prof[3*n_bins//4:].mean()
print(f'\n  Trip phases:')
print(f'    0-25%:  {fq:.1f} km/h (accel)')
print(f'    25-75%: {mh:.1f} km/h (cruise)')
print(f'    75-100%:{lq:.1f} km/h (decel)')

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
ax=axes[0]
ax.plot(pos, mn_prof, color=C[0], lw=2, label='Mean')
ax.fill_between(pos, p25p, p75p,
    color=C[0], alpha=0.2, label='IQR')
ax.fill_between(pos,
    mn_prof-sd_prof, mn_prof+sd_prof,
    color=C[0], alpha=0.08, label='1Std')
for lm in [15,20,25]:
    ax.axhline(lm, color='red',
        ls=':', lw=0.8, alpha=0.5)
ax.set_xlabel('Position (%)')
ax.set_ylabel('Speed (km/h)')
ax.set_title('Average Speed Profile')
ax.legend(fontsize=8)
ax.set_ylim(0,35)
ax=axes[1]
np.random.seed(42)
si = np.random.choice(len(prof_mat),
    size=min(50,len(prof_mat)),
    replace=False)
for i in si:
    ax.plot(pos, prof_mat[i],
        color=C[4], alpha=0.15, lw=0.5)
ax.plot(pos, mn_prof, color='black',
    lw=2.5, label='Mean')
for lm in [20,25]:
    ax.axhline(lm, color='red',
        ls=':', lw=0.8, alpha=0.5,
        label=f'{lm} km/h')
ax.set_xlabel('Position (%)')
ax.set_ylabel('Speed (km/h)')
ax.set_title('Individual Profiles (50)')
ax.legend(fontsize=8)
ax.set_ylim(0,40)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR,
    'fig08_speed_profiles.png'),
    dpi=300, bbox_inches='tight')
plt.close(fig)
print('  [Saved] fig08_speed_profiles.png')

print('\n' + '-' * 60)
print('  2c. WITHIN-TRIP VARIABILITY')
print('-' * 60)
t_stds, t_ranges, t_cvs = [], [], []
for sl in dsp['slist']:
    arr = np.array(sl, dtype=float)
    if len(arr) < 3: continue
    t_stds.append(arr.std())
    t_ranges.append(arr.max()-arr.min())
    if arr.mean() > 0:
        t_cvs.append(arr.std()/arr.mean())
t_stds=np.array(t_stds)
t_ranges=np.array(t_ranges)
t_cvs=np.array(t_cvs)
print(f'  Std: Mean={t_stds.mean():.2f},'+
      f' Med={np.median(t_stds):.2f}')
for p in [25,50,75,95]:
    print(f'    P{p}={np.percentile(t_stds,p):.2f}')
print(f'\n  Range: Mean={t_ranges.mean():.2f},'+
      f' Med={np.median(t_ranges):.2f}')
print(f'\n  CV: Mean={t_cvs.mean():.3f},'+
      f' Med={np.median(t_cvs):.3f}')

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
ax=axes[0]
ax.hist(t_stds, bins=50, color=C[0],
    alpha=0.7, edgecolor='white', lw=0.3)
ax.set_xlabel('Std Dev (km/h)')
ax.set_ylabel('Count')
ax.set_title('Speed Variability')
ax.axvline(np.median(t_stds), color='black',
    ls='--', lw=1.2,
    label=f'Med={np.median(t_stds):.1f}')
ax.legend()
ax=axes[1]
ax.hist(t_ranges, bins=50, color=C[1],
    alpha=0.7, edgecolor='white', lw=0.3)
ax.set_xlabel('Range (km/h)')
ax.set_ylabel('Count')
ax.set_title('Speed Range')
ax.axvline(np.median(t_ranges), color='black',
    ls='--', lw=1.2,
    label=f'Med={np.median(t_ranges):.1f}')
ax.legend()
ax=axes[2]
ax.hist(t_cvs[t_cvs<2], bins=50, color=C[2],
    alpha=0.7, edgecolor='white', lw=0.3)
ax.set_xlabel('CV')
ax.set_ylabel('Count')
ax.set_title('Speed CV')
ax.axvline(np.median(t_cvs), color='black',
    ls='--', lw=1.2,
    label=f'Med={np.median(t_cvs):.3f}')
ax.legend()
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR,
    'fig09_speed_variability.png'),
    dpi=300, bbox_inches='tight')
plt.close(fig)
print('  [Saved] fig09_speed_variability.png')

print('\n\n' + '=' * 72)
print('  PART 3: RIDER DEMOGRAPHICS (500K)')
print('=' * 72)

scols = ['smodel','age','seconds','start_hour']
dsc = pd.read_csv(SCOOTER,
    usecols=scols, nrows=500_000)
print(f'\nLoaded {len(dsc):,} trips.')
for c2 in ['age','seconds','start_hour']:
    dsc[c2] = pd.to_numeric(dsc[c2], errors='coerce')
print(f'\nMissing:\n{dsc.isnull().sum()}')

print('\n' + '-' * 60)
print('  3a. AGE DISTRIBUTION')
print('-' * 60)
age = dsc['age'].dropna()
age_v = age[(age>=13)&(age<=80)]
print(f'  Valid: {len(age_v):,}/{len(age):,}')
print(f'  Mean={age_v.mean():.1f},'+
      f' Med={age_v.median():.1f},'+
      f' Std={age_v.std():.1f}')
for p in pcts:
    print(f'  P{p:02d}={np.percentile(age_v,p):.0f}')

bins_a = [13,18,25,35,45,55,65,80]
labs_a = ['13-17','18-24','25-34',
          '35-44','45-54','55-64','65+']
dsc['ag'] = pd.cut(dsc['age'],
    bins=bins_a, labels=labs_a, right=False)
agc = dsc['ag'].value_counts().sort_index()
print('\n  Age groups:')
for g, cnt in agc.items():
    print(f'    {g}: {cnt:,}'+
          f' ({100*cnt/len(age_v):.1f}%)')

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
ax=axes[0]
ax.hist(age_v, bins=np.arange(13,81,1),
    color=C[0], alpha=0.7,
    edgecolor='white', lw=0.3)
ax.axvline(age_v.median(), color='black',
    ls='--', lw=1.2,
    label=f'Med={age_v.median():.0f}')
ax.set_xlabel('Age')
ax.set_ylabel('Trips')
ax.set_title('Age Distribution')
ax.legend()
ax=axes[1]
ax.bar(range(len(agc)), agc.values,
    color=C[:len(agc)], alpha=0.8,
    edgecolor='white')
ax.set_xticks(range(len(agc)))
ax.set_xticklabels(agc.index, rotation=45)
ax.set_xlabel('Age Group')
ax.set_ylabel('Trips')
ax.set_title('Trips by Age Group')
for i, v in enumerate(agc.values):
    ax.text(i, v+500, f'{v:,}',
        ha='center', fontsize=8)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR,
    'fig10_age_distribution.png'),
    dpi=300, bbox_inches='tight')
plt.close(fig)
print('  [Saved] fig10_age_distribution.png')

print('\n' + '-' * 60)
print('  3b. TRIP DURATION BY AGE')
print('-' * 60)
dsc['mins'] = dsc['seconds'] / 60
dm = (dsc['seconds']>=30) & (dsc['seconds']<=7200)
dba = dsc[dm].groupby('ag')['mins'].agg(
    ['count','mean','median','std']).round(2)
print('\n  Duration by age:')
print(dba.to_string())

fig, ax = plt.subplots(figsize=(8, 5))
vg = [g for g in labs_a
      if g in dsc[dm]['ag'].values]
dba_d = [dsc[dm & (dsc['ag']==g)]
    ['mins'].dropna().values for g in vg]
bp = ax.boxplot(dba_d, labels=vg,
    patch_artist=True, showfliers=False,
    widths=0.6,
    medianprops=dict(color='black', lw=1.5))
for ptch, clr in zip(bp['boxes'], C[:len(vg)]):
    ptch.set_facecolor(clr)
    ptch.set_alpha(0.7)
ax.set_xlabel('Age Group')
ax.set_ylabel('Duration (min)')
ax.set_title('Trip Duration by Age')
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR,
    'fig11_duration_by_age.png'),
    dpi=300, bbox_inches='tight')
plt.close(fig)
print('  [Saved] fig11_duration_by_age.png')

print('\n' + '-' * 60)
print('  3c. TIME-OF-DAY PATTERNS')
print('-' * 60)
hc = dsc['start_hour'].value_counts().sort_index()
print('\n  Trips by hour:')
for h, cnt in hc.items():
    bar = '#' * int(cnt / hc.max() * 40)
    print(f'    {int(h):02d}:00 {cnt:>7,} {bar}')
pk = hc.idxmax()
print(f'\n  Peak: {int(pk):02d}:00 ({hc.max():,})')

def tcat(h):
    if 6<=h<10: return 'Morning(6-10)'
    elif 10<=h<14: return 'Midday(10-14)'
    elif 14<=h<18: return 'Afternoon(14-18)'
    elif 18<=h<22: return 'Evening(18-22)'
    else: return 'Night(22-6)'
dsc['tc'] = dsc['start_hour'].apply(tcat)
tcc = dsc['tc'].value_counts()
print('\n  Time categories:')
for tc in ['Morning(6-10)','Midday(10-14)',
           'Afternoon(14-18)','Evening(18-22)',
           'Night(22-6)']:
    if tc in tcc:
        c2 = tcc[tc]
        print(f'    {tc}: {c2:,}'+
              f' ({100*c2/len(dsc):.1f}%)')

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
ax=axes[0]; hours = np.arange(24)
cnts = [hc.get(h, 0) for h in hours]
ax.bar(hours, cnts, color=C[0],
    alpha=0.8, edgecolor='white')
ax.set_xlabel('Hour')
ax.set_ylabel('Trips')
ax.set_title('Trip Start Time')
ax.set_xticks(hours)
ax.set_xticklabels(
    [f'{h:02d}' for h in hours],
    fontsize=7, rotation=45)
ax=axes[1]
top_ag = agc.nlargest(4).index.tolist()
for i, ag in enumerate(top_ag):
    sub = dsc[dsc['ag'] == ag]
    hourly = sub['start_hour'].value_counts()
    hourly = hourly.sort_index()
    hp = hourly / hourly.sum() * 100
    ax.plot(hp.index, hp.values, '-o',
        color=C[i], markersize=3,
        lw=1.5, label=ag)
ax.set_xlabel('Hour')
ax.set_ylabel('% Trips')
ax.set_title('Hourly by Age Group')
ax.set_xticks(hours)
ax.set_xticklabels(
    [f'{h:02d}' for h in hours],
    fontsize=7, rotation=45)
ax.legend()
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR,
    'fig12_hourly_patterns.png'),
    dpi=300, bbox_inches='tight')
plt.close(fig)
print('  [Saved] fig12_hourly_patterns.png')

print('\n' + '-' * 60)
print('  3d. VEHICLE MODELS')
print('-' * 60)
smc = dsc['smodel'].value_counts()
for m, cnt in smc.items():
    print(f'    {m}: {cnt:,}'+
          f' ({100*cnt/len(dsc):.1f}%)')

print('\n\n' + '=' * 72)
print('  SUMMARY')
print('=' * 72)
print(f'\n  SPEED (N={nt:,}):')
avgm = dv['avg_speed'].mean()
avgd = dv['avg_speed'].median()
mxm = dv['max_speed'].mean()
mxd = dv['max_speed'].median()
print(f'    avg_speed: mean={avgm:.1f}, med={avgd:.1f}')
print(f'    max_speed: mean={mxm:.1f}, med={mxd:.1f}')
e25 = (dv['max_speed']>25).mean()*100
e20 = (dv['max_speed']>20).mean()*100
print(f'    >25 km/h: {e25:.1f}%')
print(f'    >20 km/h: {e20:.1f}%')
print(f'\n  INSTANT (N={len(all_sp):,}):')
print(f'    mean={all_sp.mean():.1f},'+
      f' med={np.median(all_sp):.1f}')
ie = (all_sp>25).mean()*100
print(f'    >25 km/h: {ie:.1f}%')
print(f'\n  DEMOGRAPHICS (N={len(dsc):,}):')
print(f'    Med age: {age_v.median():.0f}')
print(f'    Peak: {int(pk):02d}:00')
print(f'    Top group: {agc.idxmax()}'+
      f' ({agc.max():,})')
print(f'\n  Figures: {OUT_DIR}')
print('\n  Files:')
for fn in sorted(os.listdir(OUT_DIR)):
    if fn.endswith('.png'):
        fp = os.path.join(OUT_DIR, fn)
        sz = os.path.getsize(fp) / 1024
        print(f'    {fn} ({sz:.0f} KB)')
print('\n' + '=' * 72)
print('  EDA COMPLETE')
print('=' * 72)

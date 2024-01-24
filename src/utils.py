"""CRM utils."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from sklearn.metrics import r2_score, mean_absolute_percentage_error
import torch

from tqdm import tqdm
import ipywidgets as widgets
from ipywidgets import interact

def in_hull(p, hull):
    """True if the point p is inside the convex hull, else otherwise."""
    if not isinstance(hull, Delaunay):
        hull = Delaunay(hull)
    return hull.find_simplex(p)>=0

def show_fields(model, coords, x, y, show_names=True, figsize=(18, 5),
                train_wells=None, test_wells=None):
    """Visualization utils."""
    locs = coords[['xn', 'yn']].values
    names = list(coords.index)
    locs_train = locs if train_wells is None else coords.loc[train_wells, ['xn', 'yn']].values

    grid = np.stack(np.meshgrid(x, y)).reshape(2, -1).T
    inhull = in_hull(grid, locs)
    grid = torch.tensor(grid).float()

    tau_field = (model.tau(grid).detach().numpy().ravel()*inhull).reshape(x.size, -1)
    J_field = (model.J(grid).detach().numpy().ravel()*inhull).reshape(x.size, -1)
    f_field = (model.f(grid).detach().numpy().ravel()*inhull).reshape(x.size, -1)

    fs = 18
    cmap = 'viridis'
    extent = (x.min(), x.max(), y.min(), y.max())

    _, ax = plt.subplots(1, 3, figsize=figsize)

    im = ax[0].imshow(tau_field, extent=extent, origin='lower', cmap=cmap, vmax=6)
    ax[0].set_title(r"Production decline time ($\tau$)", fontsize=fs)
    (cbar := plt.colorbar(im, ax=ax[0], fraction=0.046, pad=0.04)).ax.tick_params(labelsize=fs)

    im = ax[1].imshow(J_field, extent=extent, origin='lower', cmap=cmap)
    (cbar := plt.colorbar(im, ax=ax[1], fraction=0.046, pad=0.04)).ax.tick_params(labelsize=fs)
    ax[1].set_title(r'Productivity index ($J$)', fontsize=fs)

    im = ax[2].imshow(f_field, extent=extent, origin='lower', cmap=cmap)
    cbar = plt.colorbar(im, ax=ax[2], fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=fs)
    ax[2].set_title(r'Influence of injection ($f$)', fontsize=fs)

    r = max(np.ptp(x), np.ptp(y))
    for a in ax:
        a.scatter(*locs_train.T, c='r', s=30)
        if test_wells is not None:
            locs_test = coords.loc[test_wells, ['xn', 'yn']].values
            a.scatter(*locs_test.T, edgecolor='r', facecolor='none', s=30)
        if show_names:
            for i, loc in enumerate(locs):
                a.text(*(loc+0.01*r), names[i], c='r', fontsize=fs)
        a.set_xticks([])
        a.set_yticks([])
    plt.tight_layout()
    plt.show()

def get_intersection_period(df):
    """Find maximal common operating period."""
    prod = df.loc[df.group == 'P']
    rng = []
    for well, group in prod.groupby('cat'):
        tmp = group.loc[(group.liquid > 0) & (group.bhp > 0)]
        rng.append([well, tmp.date.min(), tmp.date.max()])
    inj = df.loc[df.group == 'I']
    for well, group in inj.groupby('cat'):
        tmp = group.loc[group.water_inj > 0]
        rng.append([well, tmp.date.min(), tmp.date.max()])
    rng = np.array(rng)
    dmin = rng[:, 1].max()
    dmax = rng[:, 2].min()
    if dmin > dmax:
        raise ValueError('No intersection found.')
    return dmin.date(), dmax.date()

def make_data(df, dmin=None, dmax=None, freq=None):
    """Prepare well production data."""
    wells_data = {}
    prod_names = df.loc[df.group == 'P'].cat.unique()
    inj_names = df.loc[df.group == 'I'].cat.unique()

    dmin = df.date.min() if dmin is None else pd.to_datetime(dmin)
    dmax = df.date.max() if dmax is None else pd.to_datetime(dmax)
    df = df.loc[(df.date >= dmin) & (df.date <= dmax)]

    freq = 'MS' if len(df.date.dt.day.unique()) == 1 else 'D'

    dates = pd.date_range(dmin, dmax, freq=freq)

    inject = pd.DataFrame(index=dates, columns=inj_names)
    for well in inj_names:
        data = df.loc[df.cat == well]
        inject.loc[data.date, well] = data.water_inj.values
    inject = inject.fillna(0)

    full = pd.DataFrame({'date': dates})
    for well in prod_names:
        node = full.merge(df.loc[df.cat == well], on='date', how='left').fillna(0)
        rate = node.liquid.values
        pres = node.bhp.values
        dates = node.date.values
        wells_data[well] = dict(rate=rate,
                                pres=pres,
                                dates=dates,
                                inject=inject)

    return wells_data

def get_statistics(wells_data):
    """Compute normalization values."""
    rate = np.hstack([data['rate'] for _, data in wells_data.items()])
    mean_rate = rate[rate > 0].mean()
    pres = np.hstack([data['pres'] for _, data in wells_data.items()])
    mean_pres = pres[pres > 0].mean()
    inject = np.hstack([data['inject'] for _, data in wells_data.items()])
    mean_inj = inject[inject > 0].mean()
    return dict(mean_rate=mean_rate,
                mean_pres=mean_pres,
                mean_inj=mean_inj)

def coords_scaling(coords):
    """Fit coords into 0-1 range."""
    scale = np.max([np.ptp(coords.x), np.ptp(coords.y)])
    coords[['xn', 'yn']] = (coords[['x', 'y']] - coords[['x', 'y']].min()) / scale
    return coords

def show_well(wells_data):
    """Show well's data."""
    keys = wells_data.keys()
    wells = widgets.Dropdown(options=list(zip(keys, keys)),
                             value=list(keys)[0],
                             description='Well:')
    interact(lambda well: _show_well(well, wells_data=wells_data), well=wells)

def _show_well(well, wells_data):
    """Show well's data."""
    _, ax = plt.subplots(2, 1, figsize=(8, 4))
    ax[0].plot(wells_data[well]['dates'], wells_data[well]['rate'], c='k')
    ax[0].set_ylabel('Liquid rate')
    ax[1].plot(wells_data[well]['dates'], wells_data[well]['pres'], c='g')
    ax[1].set_ylabel('BHP')

def to_tensors(wells_data, coords, stats, test_start=None):
    """Stack 1D arrays to one 2D torch tensor."""
    names = np.array(list(wells_data.keys()))

    dates = np.vstack([wells_data[k]['dates'] for k in names])
    rate_arrs = np.vstack([wells_data[k]['rate'] for k in names]) / stats['mean_rate']
    q0 = rate_arrs[:, :1]
    pres_arrs = np.vstack([wells_data[k]['pres'] for k in names]) / stats['mean_pres']
    dp = np.diff(pres_arrs, axis=1)
    inj_arrs = np.vstack([wells_data[k]['inject'].values.mean(axis=1) for k in names]) / stats['mean_inj']

    freq = wells_data[names[0]]['inject'].index.freq

    locs = np.vstack([coords.loc[k, ['xn', 'yn']].values.astype(float).ravel() for k in names])

    times = np.linspace(0, 1, dates.shape[1])

    rate_arrs = torch.tensor(rate_arrs).float()
    times = torch.tensor(times).float()
    q0 = torch.tensor(q0).float()
    inj_arrs = torch.tensor(inj_arrs).float()
    dp = torch.tensor(dp).float()
    locs = torch.tensor(locs).float()
    train_mask = torch.tensor(dates < test_start).bool() if test_start is not None else None

    dataset = dict(times=times,
                   rates=rate_arrs,
                   injs=inj_arrs,
                   dp=dp,
                   q0=q0,
                   locs=locs,
                   train_mask=train_mask,
                   _dates=dates,
                   _names=names,
                   _mean_rate=stats['mean_rate'],
                   _mean_pres=stats['mean_pres'],
                   _mean_inj=stats['mean_inj'],
                   _freq=freq)
    return dataset

def get_predictions(model, dataset):
    """Get predicted rates."""
    with torch.no_grad():
        pred = model.eval()(dataset['times'],
                            dataset['locs'],
                            dataset['dp'],
                            dataset['injs'],
                            dataset['q0'])
    return pred

def show_predictions(model, dataset, ax=None, test_start=None):
    """Make and show model's prediction and real values."""
    keys = np.unique(dataset['_names'])
    wells = widgets.Dropdown(options=list(zip(keys, keys)),
                             value=list(keys)[0],
                             description='Well:')
    interact(lambda well: _show_prediction(well, model, dataset, ax=ax, test_start=test_start),
             well=wells)

def _show_prediction(well, model, dataset, ax=None, test_start=None):
    """Make and show model's prediction and real values."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))

    preds = get_predictions(model, dataset)

    w_ind = np.where(dataset['_names'] == well)[0][0]
    dates = pd.to_datetime(dataset['_dates'][w_ind])

    preds = preds[w_ind]
    preds = np.clip(preds, 0, None) * dataset['_mean_rate']

    targs = dataset['rates'][w_ind]
    targs = targs * dataset['_mean_rate']

    ax.plot(dates, preds, label='CRM', lw=2)
    ax.plot(dates, targs, label='Real', lw=2)
    if test_start is not None:
        ax.axvline(test_start, c='gray', label='Test start')
    ax.legend()
    ax.set_title('{} MAPE={:.3f}'.format(well, mean_absolute_percentage_error(targs, preds)))

def train_model(model, n_iters, dataset, optimizer, scheduler=None, sched_step=100):
    """Training loop."""
    times = dataset['times']
    locs = dataset['locs']
    dp = dataset['dp']
    injs = dataset['injs']
    rates = dataset['rates']
    q0 = dataset['q0']
    mask = dataset['train_mask'] if 'train_mask' in dataset else None

    loss_hist = []
    for step in tqdm(range(n_iters)):
        optimizer.zero_grad()
        pred = model(times, locs, dp, injs, q0)
        sqerr = (pred - rates)**2
        loss = sqerr[mask].mean()
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            if step % sched_step == 0:
                scheduler.step()
        loss_hist.append(loss.item())
    return loss_hist

def compute_metrics(model, dataset):
    """Compute metrics."""
    preds = get_predictions(model, dataset) * dataset['_mean_rate']
    targs = dataset['rates'] * dataset['_mean_rate']

    train_mask = dataset['train_mask']
    test_mask = ~dataset['train_mask']

    wnames = dataset['_names']

    errs = []
    for well in wnames:
        wmask = np.where(dataset['_names'] == well)[0][0]

        w_train_pred = preds[wmask][train_mask[wmask]]
        w_train_targ = targs[wmask][train_mask[wmask]]

        w_test_pred = preds[wmask][test_mask[wmask]]
        w_test_targ = targs[wmask][test_mask[wmask]]

        errs.append(dict(well=well,
                         r2_train=r2_score(w_train_targ, w_train_pred),
                         mape_train=mean_absolute_percentage_error(w_train_targ, w_train_pred),
                         rmse_train=((w_train_targ-w_train_pred)**2).mean().sqrt().item(),
                         r2_test=r2_score(w_test_targ, w_test_pred),
                         mape_test=mean_absolute_percentage_error(w_test_targ, w_test_pred),
                         rmse_test=((w_test_targ-w_test_pred)**2).mean().sqrt().item()))

    return pd.DataFrame(errs).set_index('well')

def show_res_map(errs, coords, train_wells, test_wells, metrics='mape_test', lower=None, upper=None):
    """Show map of residuals and train/test split."""
    wnames = list(train_wells) + list(test_wells)

    _, ax = plt.subplots(1, 2, figsize=(10, 4))
    cbar = ax[0].scatter(coords.loc[wnames].xn, coords.loc[wnames].yn,
                         c=errs.loc[wnames][metrics].clip(lower=lower, upper=upper),
                         cmap='jet', vmin=0)
    cax = ax[0].inset_axes([.0, -0.1, 1, 0.05])
    plt.colorbar(cbar, cax=cax, orientation='horizontal')
    ax[0].set_title('MAPE for test time period')

    ax[1].scatter(coords.loc[train_wells].xn, coords.loc[train_wells].yn,
                  label='Train wells')
    ax[1].scatter(coords.loc[test_wells].xn, coords.loc[test_wells].yn,
                  label='Test wells')
    ax[1].set_title('Train/test split for wells')
    ax[1].legend(fontsize=12, loc='lower center', bbox_to_anchor=(0.5, -0.18),
                 ncol=2, fancybox=True, shadow=True)
    for a in ax:
        a.set_xticks([])
        a.set_yticks([])
    plt.show()

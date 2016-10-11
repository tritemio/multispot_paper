from pathlib import Path
import re
import matplotlib.pyplot as plt
from cycler import cycler
import seaborn as sns
import pandas as pd
import numpy as np
from scipy.stats import linregress
import lmfit

import models


blue, green, red, purple = sns.color_palette(n_colors=4)


def load_fit_data(filename):
    re_pattern = '(.+)_(.+)fit_ampl_only__window([0-9]+)s_step([0-9]+)s'
    glob_pattern = '%s*fit_ampl_only*' % filename

    params = {}
    for f in Path('results/').glob(glob_pattern):
        m = re.match(re_pattern, f.stem)
        method, window, step = m.group(2, 3, 4)
        params[method, int(window), int(step)] = pd.DataFrame.from_csv(str(f))
    return params

def load_bursts_data(filename, window, step):
    fname = 'results/%s_burst_data_vs_time__window%ds_step%ds.csv' % (filename, window, step)
    bursts = pd.DataFrame.from_csv(fname)
    return bursts

def partition_fit_data(params_all, kin, post, post2_start):
    params, params_pre, params_post, params_post2 = {}, {}, {}, {}

    for (method, window, step), pall in params_all.items():
        p = pall[(pall.tstart > kin[0]) & (pall.tstart < kin[1])].copy()
        p_pre = pall[pall.tstop < 0].copy()
        p_post = pall[(pall.tstart > post[0]) & (pall.tstart < post[1])].copy()
        p_post2 = pall[pall.tstart > post2_start].copy()

        for px in (p_pre, p_post, p_post2):
            slope, intercept, r_value, p_value, std_err = linregress(px.tstart, px.kinetics)
            y_model = px.tstart*slope + intercept
            px['kinetics_linregress'] = y_model

        params[method, window, step] = p
        params_pre[method, window, step] = p_pre
        params_post[method, window, step] = p_post
        params_post2[method, window, step] = p_post2
    return params, params_pre, params_post, params_post2

def autocorrelation(signal, t_step, delta_t_max):
    delta_t_max = min(delta_t_max, (len(signal) - 1)*t_step)
    delta_t = np.arange(0, delta_t_max, t_step, dtype=float)
    mu = signal.mean()

    corr = np.zeros_like(delta_t)
    corr[0] = ((signal - mu)*(signal - mu)).mean()
    for i in range(1, delta_t_max // t_step, 1):
        corr[i] = ((signal[i:] - mu)*(signal[:-i] - mu)).mean()
    return corr, delta_t

def _get_methods_windows_step(filename):
    re_pattern = '(.+)_(.+)fit_ampl_only__window([0-9]+)s_step([0-9]+)s'
    glob_pattern = '%s*fit_ampl_only*' % filename
    windows = set()
    methods = set()
    for f in Path('results/').glob(glob_pattern):
        m = re.match(re_pattern, f.stem)
        method, window, step = m.group(2, 3, 4)
        windows.add(int(window))
        methods.add(method)
    return sorted(list(methods)), sorted(list(windows)), int(step)

def process(filename, post, post2_start, fit=True):
    kin = [-600, post[1]]
    fig_width = 14
    fs = 18
    meth = 'em'   # by default use data generated from this fiting method
    def savefig(title, **kwargs):
        plt.savefig("figures/%s %s.png" % (filename, title))

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Load Data
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    methods, windows, step = _get_methods_windows_step(filename)
    bursts = load_bursts_data(filename, windows[1], step)
    bursts_pre = bursts[bursts.tstop < 0].copy()
    bursts_post = bursts[(bursts.tstart > post[0]) & (bursts.tstart < post[1])].copy()
    bursts_post2 = bursts[bursts.tstart > post2_start].copy()
    params_all = load_fit_data(filename)
    params, params_pre, params_post, params_post2 = partition_fit_data(params_all, kin, post, post2_start)
    p = {window: params[meth, window, step] for window in windows}

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Number of Bursts
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    fig, ax = plt.subplots(figsize=(fig_width, 3))
    ax.plot(bursts.tstart, bursts.num_bursts)
    ax.axvline(0, color='k', ls='--')
    ax.axvline(post[1], color='k', ls='--')
    ax.axvline(post2_start, color='k', ls='--')
    title = 'Number of Bursts - Full measurement'
    ax.set_title(title, fontsize=fs)
    savefig(title)

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Burst Duration
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    fig, ax = plt.subplots(figsize=(fig_width, 3))
    ax.plot(bursts.tstart, bursts.burst_width)
    ax.axvline(0, color='k', ls='--')
    ax.axvline(post[1], color='k', ls='--')
    ax.axvline(post2_start, color='k', ls='--')
    title = 'Burst Duration - Full measurement'
    ax.set_title(title, fontsize=fs)
    savefig(title)

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Number of Bursts in PRE, POST, POST2 time ranges
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    for nb, label in zip((bursts_pre, bursts_post, bursts_post2),
                         ('PRE', 'POST', 'POST2')):
        slope, intercept, r_value, p_value, std_err = linregress(nb.tstart, nb.num_bursts)
        y_model = nb.tstart*slope + intercept
        nb_corr = (nb.num_bursts - y_model) + nb.num_bursts.mean()
        nb['num_bursts_corr'] = nb_corr
        nb['linregress'] = y_model

        nbc = nb.num_bursts_corr
        nbm = nb.num_bursts.mean()
        print("%5s Number of bursts (detrended): %7.1f MEAN, %7.1f VAR, %6.3f VAR/MEAN" %
              (label, nbm, nbc.var(), nbc.var()/nbm))
        fig, ax = plt.subplots(1, 2, figsize=(fig_width, 4))
        ax[0].plot(nb.tstart, nb.num_bursts)
        ax[0].plot(nb.tstart, nb.linregress, 'r')
        ax[1].plot(nb.tstart, nb.num_bursts_corr)
        ax[1].plot(nb.tstart, np.repeat(nbm, nb.shape[0]), 'r')
        title = 'Number of bursts - %s-kinetics' % label
        fig.text(0.35, 0.95, title, fontsize=fs)
        savefig(title)

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Full Kinetic Curve (Population Fraction)
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    fig, ax = plt.subplots(figsize=(fig_width, 3))
    ax.plot('kinetics', data=params_all[meth, windows[0], step], marker='h', lw=0, color='gray', alpha=0.2)
    ax.plot('kinetics', data=params_all[meth, windows[1], step], marker='h', lw=0, alpha=0.5)
    ax.axvline(0, color='k', ls='--')
    ax.axvline(post[1], color='k', ls='--')
    ax.axvline(post2_start, color='k', ls='--')
    title = 'Population Fraction - Full measurement'
    ax.set_title(title, fontsize=fs)
    savefig(title)

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Kinetic Curve Auto-Correlation
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    w = windows[0]
    d = np.array(p[w].kinetics.loc[post[0] : post[1]])
    delta_t_max = 600  # seconds
    corr, t_corr = autocorrelation(d, t_step=step, delta_t_max=delta_t_max)

    fig, ax = plt.subplots(figsize=(fig_width, 3))
    ax.plot(t_corr, corr, '-o')
    ax.set_xlabel(r'$\Delta t$ (seconds)')
    title = 'Kinetic Curve Auto-Correlation - window = %d s' % w
    ax.set_title(title, fontsize=fs)
    savefig(title)

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Kinetic Curve in Stationary Time Ranges
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    for px, label in zip([params_pre, params_post, params_post2],  ('PRE', 'POST', 'POST2')):
        fig, ax = plt.subplots(1, 2, figsize=(fig_width, 4))
        ax[0].plot('kinetics', data=px[meth, windows[0], step], marker='h', lw=0, color='gray', alpha=0.2)
        ax[0].plot('kinetics', data=px[meth, windows[1], step], marker='h', lw=0, alpha=0.5)
        ax[0].plot('kinetics_linregress', data=px[meth, windows[1], step], color='r')
        s1, s2 = slice(None, None, windows[0]//step), slice(None, None, windows[1]//step)
        ax[1].plot(px[meth, windows[0], step].index[s1], px[meth, windows[0], step].kinetics[s1],
                   marker='h', lw=0, color='gray', alpha=0.2)
        ax[1].plot(px[meth, windows[1], step].index[s2], px[meth, windows[1], step].kinetics[s2],
                   marker='h', lw=0, alpha=1)
        ax[1].plot('kinetics_linregress', data=px[meth, windows[1], step], color='r')
        print('%5s Kinetics 30s:     %.3f STD, %.3f STD detrended.' %
              (label, (100*px[meth, windows[1], step].kinetics).std(),
               (100*px[meth, windows[1], step].kinetics_linregress).std()))
        title = 'Population Fraction - %s-kinetics' % label
        fig.text(0.40, 0.95, title, fontsize=fs)
        savefig(title)

    if not fit:
        return None, params
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Exploratory Fit
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    #method = 'nelder'
    decimation = 5

    t0_vary = False
    model = models.factory_model_exp(t0_vary=t0_vary)
    rest0f, tau = {}, {}
    for window, px in p.items():
        #____ = model.fit(np.array(px.kinetics), t=px.tstart, verbose=False, method=method)
        resx = model.fit(np.array(px.kinetics), t=px.tstart, verbose=False)
        rest0f[window] = resx
        tau[window] = resx.best_values['tau']
    tau0, tau1 = tau[windows[0]], tau[windows[1]]
    print(' FIT  Simple Exp (t0_vary=%s):  tau(w=%ds) = %.1fs  tau(w=%ds) = %.1fs  Delta = %.1f%%' %
          (t0_vary, windows[0], tau0, windows[1], tau1, 100*(tau0 - tau1)/tau0), flush=True)

    t0_vary = False
    reswt0f, tauw = {}, {}
    for window, px in p.items():
        modelw1 = models.factory_model_expwin(tau=150, t_window=window, decimation=decimation, t0_vary=t0_vary)
        #____ = modelw1.fit(np.array(px.kinetics), t=px.tstart, verbose=False, method=method)
        resx = modelw1.fit(np.array(px.kinetics), t=px.tstart, verbose=False)
        reswt0f[window] = resx
        tauw[window] = resx.best_values['tau']
    tauw0, tauw1 = tauw[windows[0]], tauw[windows[1]]
    print(' FIT  Window Exp (t0_vary=%s):  tau(w=%ds) = %.1fs  tau(w=%ds) = %.1fs  Delta = %.1f%%' %
          (t0_vary, windows[0], tauw0, windows[1], tauw1, 100*(tauw0 - tauw1)/tauw0), flush=True)

    t0_vary = True
    model = models.factory_model_exp(t0_vary=t0_vary)
    res, tau, ci = {}, {}, {}
    for window, px in p.items():
        #____ = model.fit(np.array(px.kinetics), t=px.tstart, verbose=False, method=method)
        resx = model.fit(np.array(px.kinetics), t=px.tstart, verbose=False)
        res[window] = resx
        tau[window] = resx.best_values['tau']
        ci[window] = lmfit.conf_interval(resx, resx)
    tau0, tau1 = tau[windows[0]], tau[windows[1]]
    print(' FIT  Simple Exp (t0_vary=%s):  tau(w=%ds) = %.1fs  tau(w=%ds) = %.1fs  Delta = %.1f%%' %
          (t0_vary, windows[0], tau0, windows[1], tau1, 100*(tau0 - tau1)/tau0), flush=True)

    t0_vary = True
    resw, tauw, ciw = {}, {}, {}
    for window, px in p.items():
        modelw1 = models.factory_model_expwin(tau=150, t_window=window, decimation=decimation, t0_vary=t0_vary)
        #____ = modelw1.fit(np.array(px.kinetics), t=px.tstart, verbose=False, method=method)
        resx = modelw1.fit(np.array(px.kinetics), t=px.tstart, verbose=False)
        resw[window] = resx
        tauw[window] = resx.best_values['tau']
        ciw[window] = lmfit.conf_interval(resx, resx)
    tauw0, tauw1 = tauw[windows[0]], tauw[windows[1]]
    print(' FIT  Window Exp (t0_vary=%s):  tau(w=%ds) = %.1fs  tau(w=%ds) = %.1fs  Delta = %.1f%%' %
          (t0_vary, windows[0], tauw0, windows[1], tauw1, 100*(tauw0 - tauw1)/tauw0), flush=True)

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Kinetic Curve During Transient
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    t = params[meth, windows[0], step].tstart
    fig, ax = plt.subplots(1, 2, figsize=(fig_width, 6))
    ax[0].plot('tstart', 'kinetics', data=params[meth, windows[0], step], marker='h', lw=0, color='gray', alpha=0.2)
    ax[0].plot('tstart', 'kinetics', data=params[meth, windows[1], step], marker='h', lw=0, alpha=0.5)
    ax[0].plot(t, models.expwindec_func(t, **resw[windows[1]].best_values), 'm')
    ax[0].set_xlim(kin[0], kin[1])

    s1, s2 = slice(None, None, windows[0]//step), slice(None, None, windows[1]//step)
    ax[1].plot(params[meth, windows[0], step].index[s1], params[meth, windows[0], step].kinetics[s1],
               marker='h', lw=0, color='gray', alpha=0.2)
    ax[1].plot(params[meth, windows[1], step].index[s2], params[meth, windows[1], step].kinetics[s2],
               marker='h', lw=0, alpha=1)
    ax[1].plot(t, models.expwindec_func(t, **resw[windows[1]].best_values), 'm')
    title = 'Population Fraction - kinetics'
    fig.text(0.40, 0.95, title, fontsize=fs)
    savefig(title)

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Plot Fitted Kinetic Curves
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    fitcycler = cycler('color', ('k', 'm', red))
    datacycler = cycler('color', ('grey', blue, green)) + cycler('alpha', (0.2, 0.5, 0.5))
    fig, ax = plt.subplots(2, 1, figsize=(fig_width, 8), sharex=True)

    tau_str = r'$\tau_{%ds} = %.1f s (%.1f, %.1f)$'
    for i, ((w, px), fitsty, datsty) in enumerate(zip(sorted(p.items()), fitcycler, datacycler)):
        if i == 2: break
        ax[0].plot('tstart', 'kinetics', 'o', data=px, label='', **datsty)
        label = tau_str % (w, tau[w], ci[w]['tau'][2][1], ci[w]['tau'][4][1])
        ax[0].plot(px.tstart, models.exp_func(px.tstart, **res[w].best_values),
                   label=label, **fitsty)
    ax[0].legend(loc='lower right', fontsize=fs)
    w = windows[1]
    ax[1].plot(p[w].tstart, p[w].kinetics - models.exp_func(p[w].tstart, **res[w].best_values),
               'o', color=purple)
    ax[1].set_title('Residuals - $\chi_\mathrm{red}^2 = %.4f \, 10^{-3}$' %
                    (res[w].redchi*1e3), fontsize=fs)
    ax[0].set_xlim(kin[0], kin[1])
    title = 'Kinetics Fit - Simple Exponential (t0_vary=%s)' % t0_vary
    ax[0].set_title(title, fontsize=fs)
    savefig(title)

    fig, ax = plt.subplots(2, 1, figsize=(fig_width, 8), sharex=True)
    for i, ((w, px), fitsty, datsty) in enumerate(zip(sorted(p.items()), fitcycler, datacycler)):
        if i == 2: break
        ax[0].plot('tstart', 'kinetics', 'o', data=px, label='', **datsty)
        label = tau_str % (w, tauw[w], ciw[w]['tau'][2][1], ciw[w]['tau'][4][1])
        ax[0].plot(px.tstart, models.expwindec_func(px.tstart, **resw[w].best_values),
                   label=label, **fitsty)
    ax[0].legend(loc='lower right', fontsize=fs)
    w = windows[1]
    ax[1].plot(p[w].tstart, p[w].kinetics - models.exp_func(p[w].tstart, **res[w].best_values),
               'o', color=purple)
    ax[1].set_title('Residuals - $\chi_\mathrm{red}^2 = %.4f \, 10^{-3}$' %
                    (resw[w].redchi*1e3), fontsize=fs)
    ax[0].set_xlim(kin[0], kin[1])
    title = 'Kinetics Fit - Integrated Exponential (t0_vary=%s)' % t0_vary
    ax[0].set_title(title, fontsize=fs)
    savefig(title)
    return (res, resw, rest0f, reswt0f, ci, ciw), params

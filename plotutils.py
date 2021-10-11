import torch
import numpy as np
import matplotlib.figure as figure
import matplotlib.lines as lines
from utils import *


def plot_callback(f, g, prediction, target, num_thetas, num_samples, device, writer=None, epoch=None, outfolder=None):
  thetas = torch.zeros((num_samples, 1), device=device)

  (signom, bkgnom) = prediction(thetas)
  prednom = cat(signom, bkgnom)
  transported = trans(g, signom, of_length(thetas, signom))
  transportednom = cat(transported, bkgnom)

  alltarget = target(num_samples)

  if writer is not None:
    writer.add_scalar("kldiv_nom", binned_kldiv(alltarget.detach(), transportednom.detach()), global_step=epoch)


  if num_thetas == 0:
    fig = plot_hist(
        "hist_nom"
      , epoch
      , alltarget.detach().squeeze().numpy()
      , prednom.detach().squeeze().numpy()
      , transportednom.detach().squeeze().numpy()
      )

    if outfolder is not None:
      fig.savefig(outfolder + "/hist_nom.pdf")
      pkl(fig, outfolder + "/hist_nom.pkl")

    if writer is not None:
      writer.add_figure("hist_nom", fig, global_step=epoch)

    fig.clf()

    # plot the transport vs prediction
    xval = torch.Tensor(np.mgrid[-1:0:100j]).unsqueeze(1)
    xval.requires_grad = True

    yvalnom = trans(g, xval, thetas[:100])

    fig = figure.Figure(figsize = (6, 6))
    ax = fig.add_subplot(111)

    ax.plot(detach(xval), detach(yvalnom), color = "red", lw = 2, label = "transport")

    ax.set_xlim(-1, 0)
    ax.set_ylim(-0.5, 0.5)
    ax.tick_params(axis='both', which='major', labelsize=14)

    ax.set_xlabel("$x$", fontsize=18)
    ax.set_ylabel("$x' = T(x)$", fontsize=18)
    fig.tight_layout()

    if outfolder is not None:
      fig.savefig(outfolder + "/transport_nom.pdf")
      pkl(fig, outfolder + "/transport_nom.pkl")

    if writer is not None:
      writer.add_figure("transport_nom", fig, global_step = epoch)

    fig.clf()

    # plot g vs prediction
    yvalnom = g(thetas[:100], xval)
    yvalnom = yvalnom - torch.min(yvalnom)

    fig = figure.Figure(figsize = (6, 6))
    ax = fig.add_subplot(111)

    ax.plot(detach(xval), detach(yvalnom), color = "red", lw = 2, label = "g_func")

    ax.set_xlim(-1, 0)
    ax.tick_params(axis='both', which='major', labelsize=14)

    ax.set_xlabel("$x$", fontsize=18)
    ax.set_ylabel("$g(x)$", fontsize=18)
    fig.tight_layout()

    if outfolder is not None:
      fig.savefig(outfolder + "/g_func_nom.pdf")
      pkl(fig, outfolder + "/g_func_nom.pkl")

    if writer is not None:
      writer.add_figure("g_func_nom", fig, global_step = epoch)

    fig.clf()

    xval = torch.Tensor(np.mgrid[-1:1:100j]).unsqueeze(1)
    yvalnom = f(thetas[:100], xval)
    yvalnom = yvalnom - torch.min(yvalnom)

    fig = figure.Figure(figsize = (6, 6))
    ax = fig.add_subplot(111)

    ax.plot(detach(xval), detach(yvalnom), color = "red", lw = 2, label = "f_func")

    ax.set_xlim(-1, 1)
    ax.tick_params(axis='both', which='major', labelsize=14)
    figure.xticks(ticks=[-1, -0.5, 0, 0.5, 1])

    ax.set_xlabel("$y$", fontsize=18)
    ax.set_ylabel("$f(y)$", fontsize=18)
    fig.tight_layout()

    if outfolder is not None:
      fig.savefig(outfolder + "/f_func_nom.pdf")
      pkl(fig, outfolder + "/f_func_nom.pkl")

    if writer is not None:
      writer.add_figure("f_func_nom", fig, global_step = epoch)

    fig.clf()

  else:
    for itheta in range(num_thetas):
      thetas = torch.zeros((num_samples, num_thetas), device=device)

      thetas[:,itheta] = 1
      (sigup, bkgup) = prediction(thetas)
      predup = cat(sigup, bkgup)
      transported = trans(g, sigup, of_length(thetas, sigup))
      transportedup = cat(transported, bkgup)

      thetas[:,itheta] = -1
      (sigdown, bkgdown) = prediction(thetas)
      preddown = cat(sigdown, bkgdown)
      transported = trans(g, sigdown, of_length(thetas, sigdown))
      transporteddown = cat(transported, bkgdown)

      if writer is not None:
        writer.add_scalar("kldiv_up_theta%d" % itheta, binned_kldiv(alltarget.detach(), transportedup.detach()), global_step=epoch)
        writer.add_scalar("kldiv_down_theta%d" % itheta, binned_kldiv(alltarget.detach(), transporteddown.detach()), global_step=epoch)


      fig = plot_hist(
          "hist_theta%d" % itheta
        , epoch
        , alltarget.detach().squeeze().numpy()
        , prednom.detach().squeeze().numpy()
        , transportednom.detach().squeeze().numpy()
        , predup.detach().squeeze().numpy()
        , preddown.detach().squeeze().numpy()
        , transportedup.detach().squeeze().numpy()
        , transporteddown.detach().squeeze().numpy()
        )

      if outfolder is not None:
        fig.savefig(outfolder + "/hist_theta%d.pdf" % itheta)
        pkl(fig, outfolder + "/hist_theta%d.pkl" % itheta)

      if writer is not None:
        writer.add_figure("hist_theta%d" % itheta, fig, global_step=epoch)

      fig.clf()


      xval = torch.Tensor(np.mgrid[-1:0:100j]).unsqueeze(1)
      xval.requires_grad = True

      thetas[:,:] = 0
      yvalnom = trans(g, xval, thetas[:100])

      thetas[:,itheta] = 1
      yvalup = trans(g, xval, thetas[:100])

      thetas[:,itheta] = -1
      yvaldown = trans(g, xval, thetas[:100])

      fig = figure.Figure(figsize = (6, 6))
      ax = fig.add_subplot(111)

      ax.plot(detach(xval), detach(yvalnom), color = "red", lw = 2, label = "$\\theta_%d = 0$" % itheta)
      ax.plot(detach(xval), detach(yvalup), color = "blue", lw = 2, label = "$\\theta_%d = +1$" % itheta)
      ax.plot(detach(xval), detach(yvaldown), color = "green", lw = 2, label = "$\\theta_%d = - 1$" % itheta)

      ax.plot(detach(sort(signom)), detach(sort(of_length(alltarget, signom))), "--", color = "red", lw = 2, label = "$\\theta_%d = 0$ (true)" % itheta)
      ax.plot(detach(sort(sigup)), detach(sort(of_length(alltarget, sigup))), "--", color = "blue", lw = 2, label = "$\\theta_%d = +1$ (true)" % itheta)
      ax.plot(detach(sort(sigdown)), detach(sort(of_length(alltarget, sigdown))), "--", color = "green", lw = 2, label = "$\\theta_%d = - 1$ (true)" % itheta)
      
      handles = \
        [ lines.Line2D([0], [0], color = 'black', linestyle = "dashed", label = "true optimal transport")
        , lines.Line2D([0], [0], color = 'black', linestyle = "solid", label = "derived optimal transport")
        , lines.Line2D([0], [0], marker = 's', color = 'none', markerfacecolor = 'red', markeredgecolor = 'red', label = "$\\theta_0 = 0$")
        , lines.Line2D([0], [0], marker = 's', color = 'none', markerfacecolor = 'blue', markeredgecolor = 'blue', label = "$\\theta_0 = +1$")
        , lines.Line2D([0], [0], marker = 's', color = 'none', markerfacecolor = 'green', markeredgecolor = 'green', label = "$\\theta_0 = -1$")
        ]

      fig.legend(loc=(0.4, 0.2), handles=handles, prop={'size': 14}, frameon=False)

      ax.set_xlim(-1, 0)
      ax.set_ylim(-0.5, 0.5)
      ax.tick_params(axis='both', which='major', labelsize=14)
      ax.set_xlabel("$x$", fontsize=18)
      ax.set_ylabel("$x' = T(x; \\theta_0)$", fontsize=18)
      fig.tight_layout()

      if outfolder is not None:
        fig.savefig(outfolder + "/transport_theta%d.pdf" % itheta)
        pkl(fig, outfolder + "/transport_theta%d.pkl" % itheta)

      if writer is not None:
        writer.add_figure("transport_theta%d" % itheta, fig, global_step = epoch)

      fig.clf()
    

      thetas[:,:] = 0
      yvalnom = g(thetas[:100], xval)
      yvalnom = yvalnom - torch.min(yvalnom)

      thetas[:,itheta] = 1
      yvalup = g(thetas[:100], xval)
      yvalup = yvalup - torch.min(yvalup)

      thetas[:,itheta] = -1
      yvaldown = g(thetas[:100], xval)
      yvaldown = yvaldown - torch.min(yvaldown)

      fig = figure.Figure(figsize = (6, 6))
      ax = fig.add_subplot(111)

      ax.plot(detach(xval), detach(yvalnom), color = "red", lw = 2, label = "$\\theta_%d = 0$" % itheta)
      ax.plot(detach(xval), detach(yvalup), color = "blue", lw = 2, label = "$\\theta_%d$ = +1" % itheta)
      ax.plot(detach(xval), detach(yvaldown), color = "green", lw = 2, label = "$\\theta_%d$ = - 1" % itheta)

      ax.legend(loc=(0.10, 0.75), prop={'size': 14}, frameon=False)

      ax.set_xlim(-1, 0)
      ax.tick_params(axis='both', which='major', labelsize=14)
      ax.set_xlabel("$x$", fontsize=18)
      ax.set_ylabel("$g(x; \\theta_0)$", fontsize=18)
      fig.tight_layout()

      if outfolder is not None:
        fig.savefig(outfolder + "/g_func_theta%d.pdf" % itheta)
        pkl(fig, outfolder + "/g_func_theta%d.pkl" % itheta)

      if writer is not None:
        writer.add_figure("g_func_theta%d" % itheta, fig, global_step = epoch)

      fig.clf()

      # plot f vs prediction
      xval = torch.Tensor(np.mgrid[-1:1:100j]).unsqueeze(1)
      thetas[:,:] = 0
      yvalnom = f(thetas[:100], xval)
      yvalnom = yvalnom - torch.min(yvalnom)

      thetas[:,itheta] = 1
      yvalup = f(thetas[:100], xval)
      yvalup = yvalup - torch.min(yvalup)

      thetas[:,itheta] = -1
      yvaldown = f(thetas[:100], xval)
      yvaldown = yvaldown - torch.min(yvaldown)

      fig = figure.Figure(figsize = (6, 6))
      ax = fig.add_subplot(111)

      ax.plot(detach(xval), detach(yvalnom), color = "red", lw = 2, label = "$\\theta_%d = 0$" % itheta)
      ax.plot(detach(xval), detach(yvalup), color = "blue", lw = 2, label = "$\\theta_%d$ = +1" % itheta)
      ax.plot(detach(xval), detach(yvaldown), color = "green", lw = 2, label = "$\\theta_%d$ = - 1" % itheta) 


      ax.legend(loc=(0.60, 0.75), prop={'size': 14}, frameon=False)
      ax.set_xlim(-1, 1)
      ax.tick_params(axis='both', which='major', labelsize=14)
      ax.set_xticks(ticks=[-1, -0.5, 0, 0.5, 1])

      ax.set_xlabel("$y$", fontsize=18)
      ax.set_ylabel("$f(y; \\theta_%d)$" % itheta, fontsize=18)
      fig.tight_layout()

      if outfolder is not None:
        fig.savefig(outfolder + "/f_func_theta%d.pdf" % itheta)
        pkl(fig, outfolder + "/f_func_theta%d.pkl" % itheta)

      if writer is not None:
        writer.add_figure("f_func_theta%d" % itheta, fig, global_step = epoch)

      fig.clf()

  return


def plot_hist(name, epoch, target, pred, trans, predm1=None, predp1=None, transm1=None, transp1=None):
      bins = [-1 + x*0.1 for x in range(21)]

      fig = figure.Figure(figsize=(6, 6))
      ax = fig.add_subplot(111)

      if any(x is None for x in [predm1, predp1, transm1, transp1]):
        variations = False
        hists = [ target , pred , trans ]
      else:
        variations = True
        hists = [ target , pred , trans , predm1 , predp1 , transm1 , transp1 ]

      hs, bins, _ = ax.hist( hists , bins=bins , density=True)

      (xs, htarget) = histcurve(bins, hs[0], 0)
      (xs, hprednom) = histcurve(bins, hs[1], 0)
      (xs, htransnom) = histcurve(bins, hs[2], 0)
      if variations:
        (xs, hpredup) = histcurve(bins, hs[3], 0)
        (xs, hpreddown) = histcurve(bins, hs[4], 0)
        (xs, htransup) = histcurve(bins, hs[5], 0)
        (xs, htransdown) = histcurve(bins, hs[6], 0)

      fig.clf()
      
      fig = figure.Figure(figsize=(6, 6))
      ax = fig.add_subplot(111)
      ax.scatter(
          (bins[:-1] + bins[1:]) / 2.0
        , hs[0]
        , label="target data"
        , color='black'
        , linewidth=0
        , marker='o'
        , zorder=5
        )

      ax.plot(
          xs
        , hprednom
        , color='red'
        , linewidth=2
        , linestyle="dotted"
        , label="original sim." + (" $\\theta_0 = 0$" if variations else "")
        , zorder=3
        )

      if variations:
        ax.plot(
            xs
          , hpredup
          , linewidth=2
          , color="blue"
          , linestyle="dotted"
          , label="original sim. ($\\theta_0 = +1$)"
          , zorder=3
          )

        ax.plot(
            xs
          , hpreddown
          , linewidth=2
          , color="green"
          , linestyle="dotted"
          , label="original sim. ($\\theta_0 = -1$)"
          , zorder=3
          )

      ax.plot(
          xs
        , htransnom
        , linewidth=2
        , color="red"
        , linestyle="solid"
        , label="transported sim." + (" ($\\theta_0 = 0$)" if variations else "")
        , zorder=3
        )

      if variations:
        ax.plot(
            xs
          , htransup
          , linewidth=2
          , color="blue"
          , linestyle="solid"
          , label="transported sim. ($\\theta_0 = +1$)"
          , zorder=3
          )

        ax.plot(
            xs
          , htransdown
          , linewidth=2
          , color="green"
          , linestyle="solid"
          , label="transported sim. ($\\theta_0 = - 1$)"
          , zorder=3
          )


      if not variations:
        handles = \
          [ lines.Line2D([0], [0], color = 'red', linestyle = "dotted", label = "original sim.")
          , lines.Line2D([0], [0], color = 'red', linestyle = "solid", label = "transported sim.")
          , lines.Line2D([0], [0], marker = 'o', color = 'black', label = "calibration data")
          ]

        fig.legend(loc=(0.20, 0.78), handles=handles, prop={'size': 14}, frameon=False)

      else:
        handles = \
          [ lines.Line2D([0], [0], color = 'black', linestyle = "dotted", label = "original sim.")
          , lines.Line2D([0], [0], color = 'black', linestyle = "solid", label = "transported sim.")
          , lines.Line2D([0], [0], marker = 'o', color = 'black', linewidth = 0, label = "calibration data")
          ]

        fig.legend(loc=(0.20, 0.78), handles=handles, prop={'size': 14}, frameon=False)

        handles = \
          [ lines.Line2D([0], [0], marker = 's', color = 'none', markerfacecolor = 'red', markeredgecolor = 'red', label = "$\\theta_0 = 0$")
          , lines.Line2D([0], [0], marker = 's', color = 'none', markerfacecolor = 'blue', markeredgecolor = 'blue', label = "$\\theta_0 = +1$")
          , lines.Line2D([0], [0], marker = 's', color = 'none', markerfacecolor = 'green', markeredgecolor = 'green', label = "$\\theta_0 = -1$")
          ]

        fig.legend(loc=(0.63, 0.77), handles=handles, prop={'size': 14}, frameon=False)

      ax.set_ylim(0, max(htarget)*1.5)
      ax.set_xlim(-1, 1)
      ax.set_xticks(ticks=[-1, -0.5, 0, 0.5, 1])
      ax.tick_params(axis='both', which='major', labelsize=14)
      ax.set_xlabel("$x$, $x'$, $y$", fontsize=18)
      ax.set_ylabel("binned probability density", fontsize=18)
      fig.tight_layout()

      return fig
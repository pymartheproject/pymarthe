"""
Contains some helper functions for Marthe/PEST model postprocessing

"""


# -- Import global modules
import os, sys
import numpy as np
import pandas as pd
import pyemu
from copy import deepcopy


# -- Import plot modules
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pymarthe.utils import marthe_utils, shp_utils, pest_utils, pp_utils







class PestPostProcessing():
    """
    Helper class to manage some post-processing operation
    (especially for plotting purpose).
    """
    def __init__(self, pst):
        """

        Parameters
        -----------
        pst (str/pyemu.Pst) : Pest Control File.
                              If string, the `pyemu.Pst(pst)` will be performed
                              to fetch the pyemu pst instance.

        Examples
        -----------
        ppp = PestPostProcessing('mypst.pst')

        """
        self.pst = pyemu.Pst(pst) if isinstance(pst, str) else pst
        self.hasreg = True if self.pst.control_data.pestmode == 'regularization' else False



    def phi_progress(self, pest_exe='++', log=True, phimlim=False, phimaccept=False, filename=None):
        """
        Plot the objective function (phi) evolution.

        Parameters
        -----------
        pest_exe (str, optional) : pest executable used.
                                   Can be:
                                    - '++' : pestpp (-glm, -opt ,-swp, ...)
                                    - 'hp' : pesthp
                                   Default is '++'.

        log (bool, optional) : whatever switch phi values to logaritmic scale (base 10).
                               Default is True.

        phimlim (bool, optional) : whatever plot a horizontal line at phimlim value.
                                   Default is False.

        phimaccept (bool, optional) : whatever plot a horizontal line at phimaccept value.
                                      Default is False.

        filename (str, optional): output file name to save plot on disk.
                                  If None, the plot will not be saving on disk.
                                  Default is None.

        Returns:
        --------
        ax (AxeSubplot) : plot axe.

        Examples
        -----------
        ppp = PestPostProcessing('mypst.pst')
        ppp.phi_progress(phimlim=True, filename='phi_progress.png')

        """
        # ---- Get phi values according to pest output file(s)
        if pest_exe == 'hp':
            # -- Get ofr file
            ofr_file = self.pst.filename.replace(".pst",".ofr")
            # -- Load ofr data as dataframe
            df = pd.read_csv(ofr_file, skiprows = list(range(3)), sep = r'\s+', index_col = False)
            # -- Extract usefull data for plot
            it, phi, reg_phi = df.iteration, df.measurement, df.regularisation
        else:
            # -- Get iobj file
            iobj_file = self.pst.filename.replace(".pst",".iobj")
            # -- Load iobj data as dataframe
            df = pd.read_csv(iobj_file)
            # -- Extract usefull data for plot
            it, phi, reg_phi = df.iteration, df.measurement_phi, df.regularization_phi
        # ---- Prepare plot figure
        plt.figure(figsize=(9,6))
        plt.rc('font', family='serif', size=10)
        ax = plt.subplot(1,1,1)
        # ---- Plot Phi measured
        a, = ax.plot(it, phi, color='tab:blue', marker='.', label='$\Phi_{measured}$')
        ax.set_xticks(it, [f'it{i}' for i in it])
        # ---- Plot phi regularization if exists
        if self.hasreg:
            ax1=ax.twinx()
            b, = ax1.plot(it, reg_phi, color='tab:orange', marker='+', label='$\Phi_{regul}$')
            ax1.set_ylabel('Regularization objective function ($\Phi_r$)', color='tab:orange')
            # -- Add PHIMLIM
            if phimlim:
                ax.hlines(pst.reg_data.phimlim, 0, len(it)-1, 
                          colors='navy', linestyles='solid', label='$\Phi_{limit}$')
            # -- Add PHIMACCEPT
            if phimaccept:
                ax.hlines(pst.reg_data.phimaccept, 0, len(it)-1, 
                          colors='darkblue', linestyles='dotted', label='$\Phi_{accept}$')
        # ---- Add log scale if required
        if log:
            ax.set_yscale('log', base= 10)
        # ---- Set labels
        ax.set_xlabel('Iterations')
        ax.set_ylabel('Measurement objective function ($\Phi_m$)', color='tab:blue')
        
        # ---- Set grid
        ax.grid(color='lightgrey')
        # ---- Set legend
        leg_data = [[a,b], [p_.get_label() for p_ in [a,b]]] if self.hasreg else [[a], [a.get_label()]]
        plt.legend(*leg_data, loc= 'upper center')
        plt.tight_layout()
        # ---- Export plot if required
        if filename is not None:
            plt.savefig(filename)
        # ---- Return plot Axe
        return ax




    def phi_components(self, obs_groups=None, explode = 0.05, filename = None):
        """
        Plot the contribution of each observation group to the total objective function.

        Parameters
        -----------
        obs_groups (str/list[str], optional) : required observation groups to consider.
                                               If None, all observation groups will be considered.
                                               Default is None.

        explode (float, optional) : space between pie chart parts.
                                    Default is 0.05.

        filename (str, optional): output file name to save plot on disk.
                                  If None, the plot will not be saving on disk.
                                  Default is None.

        Returns:
        --------
        ax (AxeSubplot) : plot axe.

        Examples
        -----------
        ppp = PestPostProcessing('mypst.pst')
        ppp.phi_components(filename='phi_components.png')

        """
        # ---- Manages observation group names
        obs_groups = self.pst.obs_groups if obs_groups is None else marthe_utils.make_iterable(obs_groups)
        # ---- Fetch phi components data from pst
        pcompo_dic = {key : self.pst.phi_components[key] for key in obs_groups}
        # ---- Get component labels & values
        labels = list(pcompo_dic.keys())
        n_labels = len(labels)
        sizes = list(pcompo_dic.values())
        # ---- Generate colors for each component
        colors = [ np.random.rand(3,) for i in range(n_labels) ]
        # ---- Distance between pie parts
        explode = tuple(np.ones(n_labels)*explode)
        # ---- Prepare ploting zone
        plt.figure(figsize=(9,9))
        ax = plt.subplot()
        plt.rc('font', family='serif', size=9)
        # ---- Plot pie chart
        ax.pie(sizes, colors = colors, labels=labels, autopct='%1.1f%%',
               startangle=90, pctdistance=0.75, explode = explode)
        # ---- Draw circle
        centre_circle = plt.Circle((0,0),0.60,fc='white')
        fig = plt.gcf()
        fig.gca().add_artist(centre_circle)
        # ---- Set visualisation parameters
        ax.axis('equal')  # Equal aspect ratio ensures that is drawn as a circle
        plt.tight_layout()
        # ---- Export plot if required
        if filename is not None:
            plt.savefig(filename)
        # ---- Return Axe
        return ax



    def export_stats(self, filename, cmap='YlGnBu', engine='excel'):
        """
        Export/save some pest criterias (rss, mae, rmse, nrmse) 
        for each observation groups as data table.

        Parameters
        -----------
        filename (str): output file name to save data table on disk.

        cmap (str, optional) : matplotlib continous color palette to 
                               use on table values.
                               Default is 'YlGnBu'.

        engine (str, optional) : required exporting provider.
                                 Can be:
                                    - 'excel' ('.xlsx')
                                    - 'latex' ('.tex')
                                    - 'html'  ('.html')
                                Default is 'excel'.

        Returns:
        --------


        Examples
        -----------
        ppp = PestPostProcessing('mypst.pst')
        ppp.export_stats('stats.xlsx', cmap='viridis')

        """
        # ---- Get pest statistics DataFrame (criterias)
        df = self.pst.get_res_stats().T
        # ---- Get DataFrame styler
        first_row_properties = {'background-color': 'white', 'font-size': '13pt', 'color': 'red',
                               'border-color': 'red', 'border-style' :'dashed', 'border-width': '2px'}
        styler = df.style.set_caption("PEST ANALYSIS CRITERIAS"
                        ).background_gradient(cmap=cmap, axis=0,
                                              subset= pd.IndexSlice[df.index.drop('all'), :]
                        ).set_properties(**{'font-size': '11pt'}
                        ).set_properties(subset = pd.IndexSlice['all', :], **first_row_properties)
        # ---- Export according to engine
        if engine == 'excel':
            styler.to_excel(filename)
        elif engine == 'latex':
            styler.to_latex(filename)
        elif engine == 'html':
            styler.to_html(filename)




    def obs_vs_sim(self, kind='1to1', reifile=None, obs_groups=None, figsize=None, onefile=True, path=None):
        """
        Graphical analysis of observed vs simulated values.

        Parameters
        -----------
        kind (str, optional) : kind of plot required.
                               Can be :
                                  - '1to1' : (x=measured, y=modelled)
                                  - 'record' : (x=observations, y=measured & modelled)
                                  - 'residual' : (x=observations, y=measured - modelled)
                               Default is '1to1'.

        reifile (str, optional) : pest residual output file.
                                  Format: 'pst_name' + '.rei' + 'n°itération' (ex: 'cal.rei5')
                                  If None, the `.res` attriut in input pst wil be considered,
                                  generally it refers to the last iteration.
                                  Default is None.

        obs_groups (str/list[str], optional) : required observations groups.
                                               If None, all observation groups with at least 
                                               one not null observation will be considered.
                                               Default is None.
        
        figsize (tup, optional) : plot figure size (height, width).
                                  Default is:
                                    - '1to1' : (6,6)
                                    - 'record' : (9,5)
                                    - 'residual' : (9,4)

        onefile (bool, optional) : whatever export plots in a unique pdf file.
                                   Default is True.

        path (str, optional): whatever export the produced plots.
                              If None, plots will not be saved.
                              Otherwise:
                                If `onefile` is True: output file name to save pdf file on disk.
                                If `onefile` is False: folder path to save plots on disk.
                              Default is None.

        Returns:
        --------
        axs (dict) : dictionary of all plots (AxesSuplot).
                     Format: {loc_id: ax} (ex: {'well_03': <AxesSubplot>})

        Examples
        -----------
        ppp = PestPostProcessing('mypst.pst')

        ppp.obs_vs_sim(kind='record', reifile='cal_r_lizonne.rei9',
                      obs_groups=[obg if obg.startswith('p') for obg in ppp.nnz_obs_group],
                      onefile=True, path='obs_vs_sim_records.pdf')

        ppp.obs_vs_sim(kind='1to1', reifile='cal_r_lizonne.rei13',
                       onefile=False, path='pest/post_proc')

        """
        # ---- Manage required observation groups
        obgnmes = self.pst.nnz_obs_groups if obs_groups is None else marthe_utils.make_iterable(obs_groups)

        # ---- Get/set résiduals values
        if reifile is not None:
            self.pst.set_res(reifile)
        res = self.pst.res.query("group in @obgnmes")
        res[['loc_id', 'obs_id']] = res['name'].str.extract("loc(\d{3})n(\d+)")
        res.index = res['obs_id'].apply(lambda s: f'obs_{s}')

        # ---- Build plots according to required kind of plot
        axs = {}

        # -- 1to1 
        if kind == '1to1':
            # -- Iterate over locnames
            for loc_id, df in res.groupby('loc_id'):
                # -- Get observation group_name
                gname = df.loc[df.loc_id == loc_id]['group'][0]
                # -- Prepare plot figure
                fs = (6,6) if figsize is None else figsize 
                plt.figure(figsize=fs)
                ax = plt.subplot()
                # -- Plot scatter obs vs sim
                ax = df.plot('measured', 'modelled', ax=ax, lw=0,
                              marker='+', color='navy', ms=4, mew=0.7,
                              alpha=0.7, legend=False, zorder=50)
                # -- Fetch plot min/max xy-limits
                limits = [*ax.get_xlim(), *ax.get_ylim()]
                xylimits = [min(limits[::2]), max(limits[1::2])]
                # -- Re-scale plot
                ax.set_xlim(xylimits)
                ax.set_ylim(xylimits)
                # -- Plot 1:1 line
                ax.plot(*[xylimits]*2, lw=1, color='black', zorder=10)
                # -- Set title
                ax.set_title(f'Group: {gname}, Location: loc{loc_id}, kind: {kind}',
                             fontsize=10, fontweight='bold')
                # -- Set xy-labels with descent size
                ax.set_xlabel('Measured', fontsize=11)
                ax.set_ylabel('Modelled', fontsize=11)
                # -- Add background grid
                ax.grid('lightgrey', lw=0.5, zorder=0)
                # -- Store current AxeSubplot in dictionary
                axs[loc_id] = ax


        elif kind == 'record':
            # -- Iterate over locnames
            for loc_id, df in res.groupby('loc_id'):
                # -- Get observation group_name
                gname = df.loc[df.loc_id == loc_id]['group'][0]
                # -- Prepare plot figure
                fs = (9,5) if figsize is None else figsize 
                plt.figure(figsize=fs)
                ax = plt.subplot()
                # -- Plot simulated values as red line
                df['modelled'].plot(figsize=figsize, color='red', lw=1)
                # -- Plot observed values as blue cross markers
                df['measured'].plot(ax=ax, color='blue', lw=0, marker='+', ms=3, mew=0.5)
                # -- Set title
                ax.set_title(f'Group: {gname}, Location: loc{loc_id}, kind: {kind}',
                             fontsize=10, fontweight='bold')
                # -- Add some usefull information in top left corner
                rmse = np.sqrt(((df['modelled'] - df['measured']) ** 2).mean())
                text = '\n'.join( [ f'Number of observation: {len(df)}',
                                    f'Max residual: {round(df["residual"].max(),5)}',
                                    f'RMSE: {round(rmse,5)}']
                                )
                ax.text(0.01, 0.98, text, va='top',
                                           linespacing =1.5,
                                           fontfamily='serif',
                                           fontsize = 7,
                                           transform=ax.transAxes)
                # -- Add legend
                ax.legend(fontsize=10)
                # -- Remove x-label
                ax.set_xlabel('')
                # -- Store current AxeSubplot in dictionary
                axs[loc_id] = ax

        elif kind == 'residual':
            # -- Iterate over locnames
            for loc_id, df in res.groupby('loc_id'):
                # -- Get observation group_name
                gname = df.loc[df.loc_id == loc_id]['group'][0]
                # -- Prepare plot figure
                fs = (9,4) if figsize is None else figsize 
                plt.figure(figsize=fs)
                ax = plt.subplot()
                # -- Bar plot of residuals with binary color (<0 red, >0 blue)
                df['residual'].plot.bar(ax=ax, alpha=0.5, width=1,
                        color= (df.residual > 0).map({True:'b', False:'g'}))
                # -- Add base line reisidual =0
                ax.axhline(y=0, color='black', lw=0.5)
                # -- Plot either min/max median for both negative/positive residuals
                ax.axhline(y=df.residual.mean(),  color='r', lw=0.5, ls='--')
                # -- Clean x axis (ticks, labels, ..)
                ax.set_xlabel('')
                ax.set_xticklabels([])
                ax.set_xticks([])
                # -- Add title
                ax.set_title(f'Group: {gname}, Location: loc{loc_id}, kind: {kind}',
                             fontsize=10, fontweight='bold')
                # -- Store current AxeSubplot in dictionary
                axs[loc_id] = ax

        # ---- Manage export
        if path is not None:
            # -- To unique file
            if onefile:
                self.to_onefile(axs.values(), path)
            # -- To multiple files
            else:
                for loc_id, ax in axs.items():
                    # -- Fetch observation group name
                    group = res.loc[res.loc_id == loc_id]['group'][0]
                    # -- Build output file name
                    fn = os.path.join(path, f'{group}_loc{loc_id}.png')
                    # -- Wrirte on disk
                    ax.get_figure().savefig(fn)

        # ---- Return plots
        return {int(k):v for k,v in axs.items()} 




    def get_css_df(self, parnames=None):
        """
        Read Composite Scaled Sensitivities (CSS) as pandas DataFrame.

        Parameters
        -----------
        parnames (str/list[str], optional) : required parameters full name(s).
                                             If None, all adjustable parameters will be considred.
                                             Default is None.

        Returns:
        --------
        css_df (DataFrame) : composite scaled sensitivities.
                             Format:

                                                  param1     ..       paramN
                            iteration
                                    1       7.733360e-08     ..    2.357000e-08
                                    2       5.316330e-08           1.575150e-07
                                    3       5.066230e-08     ..    5.618370e-09
                                   ..                 ..                     ..
                                    N       9.974170e-05     ..    4.767230e-05


        Examples
        -----------
        ppp = PestPostProcessing('mypst.pst')
        css_df = ppp.get_css_df()
        """
        # ---- Fetch `.isen` file from pst filename
        senfile = self.pst.filename.replace(".pst",".isen")
        # ---- Manage required parameters names to include
        pn = self.pst.adj_par_names if parnames is None else marthe_utils.make_iterable(parnames)
        # ---- Read and subset
        css_df = pd.read_csv(senfile, index_col=0).loc[:,pn]
        # ---- Return as DataFrame
        return css_df





    def export_css(self, filename, parnames=None, cmap='YlGnBu', engine='excel'):
        """
        Export Composite Scaled Sensitivities (CSS).

        Parameters
        -----------
        filename (str): output file name to save css data on disk.

        cmap (str, optional) : matplotlib continous color palette to 
                               use on table values.
                               Default is 'YlGnBu'.

        engine (str, optional) : required exporting provider.
                                 Can be:
                                    - 'excel' ('.xlsx')
                                    - 'latex' ('.tex')
                                    - 'html'  ('.html')
                                Default is 'excel'.

        Returns:
        --------


        Examples
        -----------
        ppp = PestPostProcessing('mypst.pst')
        ppp.export_css('css.xlsx')

        """
        # ---- Fetch css DataFrame
        css_df = self.get_css_df(parnames=parnames)
        # ---- Build style
        styler = css_df.style.set_caption("COMPOSITE SCALED SENSITIVITIES (CSS)"
                            ).background_gradient(cmap=cmap, axis=0
                            ).set_properties(**{'font-size': '11pt'})
        # ---- Export according to engine
        if engine == 'excel':
            styler.to_excel(filename)
        elif engine == 'latex':
            styler.to_latex(filename)
        elif engine == 'html':
            styler.to_html(filename)





    @staticmethod
    def to_onefile(axs, filename):
        """
        Combine multiple plots into a unique pdf file.

        Parameters
        -----------
        axs (list) : set of unique AxesSubplot to store in file.

        filename (str): output pdf file name to save plots on disk.
                        Note: must have the '.pdf' extension.

        Returns:
        --------


        Examples
        -----------
        df = pd.DataFrame(data={'r':np.linspace(5,58,50)})
        axs = [df.plot(), (df*df).plot(), df.add(22).plot()]
        PestPostProcessing.to_onefile(axs, 'all_plots.pdf')

        """ 
        # ---- Open global pdf
        with PdfPages(filename) as pdf:
            # -- Iterate over axes
            for ax in axs:
                pdf.savefig(ax.get_figure())




    def parameter_progress(self, parnames=None, pargroups=None, pest_exe='++', figsize=None, onefile=True, path=None):
        """
        Graphical analysis of the parameters values evolution along 
        iteration from PEST calibration process.

        Parameters
        -----------
        parnames (str/list[str], optional) : required parameters full name(s).
                                             If None, all adjustable parameters will be considred.
                                              Default is None.

        pargroups (str/list[str], optional) : required parameters group name(s).
                                              If None, all adjustable parameter groups will be considred.
                                              Default is None.

        pest_exe (str, optional) : pest executable used.
                                   Can be:
                                    - '++' : pestpp (-glm, -opt ,-swp, ...)
                                    - 'hp' : pesthp
                                   Default is '++'.

        figsize (tup, optional) : plot figure size (height, width).
                                  Default is (9,4).

        onefile (bool, optional) : whatever export plots in a unique pdf file.
                                   Default is True.

        path (str, optional): whatever export the produced plots.
                              If None, plots will not be saved.
                              Otherwise:
                                If `onefile` is True: output file name to save pdf file on disk.
                                If `onefile` is False: folder path to save plots on disk.
                              Default is None.

        Returns:
        --------
        axs (dict) : dictionary of all plots (AxesSuplot).
                     Format: {parnme: ax} (ex: {'hk_zpc_l11_z01': <AxesSubplot>})

        Examples
        -----------
        ppp = PestPostProcessing('mypst.pst')

        ppp.parameter_progress(pargroups='soil', onefile=True,
                               path='soil_params_evolution.pdf')

        ppp.obs_vs_sim(parnames= [p if p.startswith('cap_sol') for p in ppp.pst.adj_par_names()]
                       onefile=False, path='pest/post_proc')

        """
        # ---- Manage required parameters names and groups
        pg = self.pst.adj_par_groups if pargroups is None else marthe_utils.make_iterable(pargroups)
        pn = self.pst.adj_par_names if parnames is None else marthe_utils.make_iterable(parnames)

        # ---- Extract parameter evolution through pest iteration
        if pest_exe == '++':
            # ---- Create dataframe with initial parameters + PEST iterations
            df_ipar = pd.read_csv(self.pst.filename.replace('.pst','.ipar'), index_col=0)
            df_ipar.index = ['it' + str(n_it) for n_it in df_ipar.index]

        elif pest_exe == 'hp':
                    # ---- Get ofr filename
            ofr_file = self.pst.filename.replace(".pst",".ofr")
            # ---- Load ofr data as dataframe
            ofr_df = pd.read_csv(ofr_file, skiprows = list(range(3)), sep = r'\s+', index_col = False)
            # ---- Fetch number of iteration
            n_it = len(ofr_df) - 1
            # ---- Collect parameter values over iterations
            ipar_list = []
            for it in range(n_it):
                par_df = pd.read_csv(self.pst.filename.replace('.pst', '.par.{}'.format(str(it + 1))),
                                     skiprows = [0], usecols = [1],
                                     header = None, sep = r'\s+', index_col = False)
                par_df.set_index(self.pst.parameter_data.parnme, inplace = True)
                par_df.columns = ['it' + str(it + 1)]
                ipar_list.append(par_df.T)
                # ---- Concat all values
                df_ipar = pd.concat(ipar_list)

        # ---- Subset only required parameters
        q = "pargp in @pg & parnme in @pn & partrans not in ['fixed', 'tied']"
        df = df_ipar[self.pst.parameter_data.query(q).index]

        # ---- Plot all required parameters
        axs = {}
        for p in df.columns:
            # -- Prepare plot area
            fs = (9,4) if figsize is None else figsize 
            ax = df.plot(y=p,  marker='o', linestyle='-', figsize=fs, lw=1,
                         ms=4, color='black', zorder=20, legend=False)
            # -- Set ticks for all iterations
            xticks = np.arange(len(df))
            ax.set_xticks(xticks, df.index)
            # -- Add colored vertical areas
            for coo in xticks[::2]:
                if coo > 0:
                    ax.axvspan(coo, coo-1 , facecolor='burlywood', alpha=0.5)
                ax.axvspan(coo, coo+1 , facecolor='bisque', alpha=0.5)
            # -- Add grid vertical lines
            ax.grid(axis='y', color='lightgrey', lw=0.3)
            # -- Get variation between isteps as string
            sign = (df[p].diff().dropna() > 0).map({True: '+', False: ''})
            vvar = sign + df[p].diff().dropna().round(4).astype(str)
            # -- Get position of variation values to plot
            xvar = xticks[:-1] + 0.5
            yvar = [df[p].min()]*len(xvar)
            # -- Set variation text bounding box properties
            bbox = dict(boxstyle='round4,pad=0.4', fc='black', lw=0, zorder=50)
            # -- Add variation values on plot
            for x,y,v in zip(xvar, yvar, vvar):
                ax.text(x,y,v, color='white', fontweight='bold', fontsize=5,
                        va='center', ha='center', bbox=bbox, zorder=50)
            # -- Add title with parameter name
            ax.set_title(f'Evolution of parameter: {p}',fontsize=10, fontweight='bold')
            # -- Store current AxeSubplot in dictionary
            axs[p] = ax

        # ---- Manage export
        if path is not None:
            # -- To unique file
            if onefile:
                self.to_onefile(axs.values(), path)
            # -- To multiple files
            else:
                for p, ax in axs.items():
                    # -- Build output file name
                    fn = os.path.join(path, f'{p}_progress.png')
                    # -- Wrirte on disk
                    ax.get_figure().savefig(fn)

        # ---- Return plots
        return axs 


    def __str__(self):
        """
        Internal string method.
        """
        return 'PestPostProcessing'







# def get_heads(mm, simfile=None, as_array=False):
#     """
#     """
#     mfs = MartheFieldSeries(mm, field='charge', simfile=simfile)
#     if as_array:
#         heads = np.squeeze( [ mf.data['value'].reshape(
#                                 (mm.nlay, mm.ncpl)) 
#                                     for i,mf in mfs.data.items()]
#                                     )
#         return heads
#     else:
#         return mfs




# def get_gradients(mm, heads=None, simfile=None, istep=None, masked_values = [9999,8888,-9999], as_array=False):
#     """
#     """
#     # -- Fetch head data as 3D-array (steps, layers, cells)
#     heads =  get_heads(mm, simfile, as_array=True) if heads is None else heads

#     # -- Get head data as masked array
#     hds = np.ma.array(heads, ndmin = 3, mask = np.isin(heads, masked_values))

#     # -- Manage istep to perform
#     _istep = list(range(mm.nstep)) if istep is None else marthe_utils.make_iterable(istep)

#     # -- Get top of each layer
#     try:
#         from pymarthe.utils.vtk_utils import get_top_botm
#     except:
#         ImportError('Could not import pymarthe.utils.vtk_utils.get_top_botm().')

#     top, botm = get_top_botm(mm, hws=mm.hws)


#     grad = []
#     for istep in _istep:
#         # -- Get head for istep
#         h = hds[istep]
#         # -- Set head values on unsaturated zone (unsat = z > h)
#         z = np.ma.array(top, mask=h.mask)
#         z[z > h] = h[z > h]

#         # ---- Apply .diff on data and mask components separately
#         diff_mask = np.diff(h.mask, axis=0)
#         dz = np.ma.array(np.diff(z.data, axis=0), mask=diff_mask)
#         dh = np.ma.array(np.diff(h.data, axis=0), mask=diff_mask)
#         # convert to 9999-filled array
#         g = np.concatenate( [ (dh / dz).filled(9999),
#                                np.ones((1, mm.ncpl))*9999 ]
#                                )
#         grad.append(g)

#     if as_array:
#         return np.squeeze(grad)
#     else:
#         ndig = len(str(len(grad)))
#         grad_fields= {}
#         for i, istep in enumerate(_istep):
#             rec = deepcopy(mm.imask.data)
#             rec['value'] = np.ravel(grad[i,:,:])
#             grad_fields[istep] = MartheField(field= 'gradient_' + str(i).zfill(ndig),
#                                              data=rec,
#                                              mm=mm)
#         return grad_fields



# # test
# mfs = get_heads(mm, simfile='chasim_cal_histo.out', as_array=False)
# heads = get_heads(mm, simfile='chasim_cal_histo.out', as_array=True)
# gradients = get_gradients(mm, heads = heads, as_array=True)
# grad_dic = get_gradients(mm, heads = heads, as_array=False)



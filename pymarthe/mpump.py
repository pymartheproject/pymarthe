"""
Contains the MarthePump class (Subclass of MartheModel)
for handling pumping conditions by locations.
"""

import os
import numpy as np
import pandas as pd
import re, ast
from .utils import marthe_utils, pest_utils
import warnings

encoding = 'latin-1'


class MarthePump():
    """
    Class for handling Marthe pumping data.

    """

    def __init__(self, mm, pastp_file = None,  mode = 'aquifer', verbose=False):
        """
        MarthePump class : available for aquifer or river pumping.

        Parameters
        ----------
        mm (MartheModel): parent model.
        pastp_file (str,optional) : name (or full path) of the .pastp file.
                                    Default: taken from parent model.
        mode (str,optional) : type of  pumping.
                              Can be 'aquifer' or 'river'.
                              Default is 'aquifer'.

        Examples
        --------
        mm = MartheModel(rma_file)
        mpump = MarthePump(mm, mode = 'aquifer')
        """
        # ---- Pointer to parent model
        self.mm = mm
        self.nstep = self.mm.nstep

        # ---- Stock pumping type as attribute
        self.mode = mode
        self.prop_name = 'aqpump' if self.mode == 'aquifer' else 'rivpump'

        # ---- Fetch pastp name if not provided
        self.pastp_file = self.mm.mlfiles['pastp'] if pastp_file is None else pastp_file

        # ---- Available/supported variables and qtypes
        self.vars = ['istep', 'node', 'layer', 'i', 'j', 'value', 'boundname']
        self._vars = self.vars + ['qfilename', 'qtype', 'qrow', 'qcol']
        self.qtypes = ['mail', 'record', 'listm']

        # ---- Read pumping data (and metadata) from .pastp file according to pumping type (DataFrame)
        self._extract_data(mode)
        # ---- Set property style
        self._proptype = 'list'
        # ---- Perform some validity check if required (raise warnings)
        if verbose:
            self._verbose()





    def _verbose(self):
        """
        Perform some classic checks about pumping data validity and print
        bad behaviour as warnings
            - Search for pumpings in inactive cells ()
            - Search for multiple pumping on same cell 

        """
        # -- Search for pumping data on inactive cells
        warn_msg = "Pumping condition applied on inactive cell : "
        try:
            for node in self.data['node']:
                if self.mm.imask.data['value'][node] == 0:
                    coords = 'Node = {}, Layer = {}, Nested = {}, Row = {}, Column = {} .'.format(
                                    node,
                                    self.mm.imask.data['layer'][node],
                                    self.mm.imask.data['inest'][node],
                                    self.mm.imask.data['i'][node],
                                    self.mm.imask.data['j'][node]
                                )
                    warnings.warn(warn_msg + coords)
        except:
            pass



        # -- Search for several pumping data on same node
        warn_msg = "Multiple pumping condition in same cell : "
        try:
            agg = self.data.groupby(['istep','node'], as_index=False).size()
            nodes = agg.loc[agg['size'] > 1, 'node'].unique()
            for node in nodes:
                coords = 'Node = {}, Layer = {}, Nested = {}, Row = {}, Column = {} .'.format(
                                node,
                                self.mm.imask.data['layer'][node],
                                self.mm.imask.data['inest'][node],
                                self.mm.imask.data['i'][node],
                                self.mm.imask.data['j'][node]
                            )
                warnings.warn(warn_msg + coords)
        except:
            pass




    def _extract_data(self, mode):
        """
        Extract pumping data from .past file according to the pumping mode.
        Wrapper to marthe_utils.extract_pastp_pumping()

        Parameters:
        ----------
        mode (str) : pumping mode.
                     Can be 'aquifer' or 'river'.

        Returns:
        --------
        Set pumping data and meta data inplace (as attribute)

        Examples:
        --------
        mp._extract_data()

        """
        
        # ---- Manage 'aqpump'
        if self.mode == 'aquifer':
            d, _d = marthe_utils.extract_pastp_pumping(self.pastp_file, mode)

        # ---- Manage 'rivpump'
        elif self.mode == 'river':
            # ---- Convert aff/trc data in column, line, plan (layer) format in .pastp file
            marthe_utils.convert_at2clp(self.pastp_file, mm = self.mm)
            d, _d = marthe_utils.extract_pastp_pumping(self.pastp_file, mode)

        # -- Manage xy inputs
        if all(loc in d.columns for loc in list('xy')):
            # -- Print message to inform about convertion
            print('Converting xy pumping data into row(s), column(s) ...')
            # -- Get all nodes
            nodes = self.mm.get_node(x=_d.x,y=_d.y,layer=_d.layer)
            # -- Perform query on modlegrid
            _d[['node','i','j']] = self.mm.query_grid(node=nodes, target=['i','j']).reset_index()
            # -- Push to class attribute
            self.data, self._data = _d[self.vars], _d[self._vars + ['x','y']]

        # -- Manage ij inputs
        else:
            # -- Perform query on modelgrid
            _d['node'] = self.mm.query_grid(i=_d.i, j=_d.j, layer=_d.layer).reset_index()['node']
            # -- Push to class attribute
            self.data, self._data = _d[self.vars], _d[self._vars]


        # ---- Set generic boundnames
        digits = len(str(self.data.node.max()))
        bdnmes =  ['{}_{}'.format(
                        self.prop_name,
                        str(node).zfill(digits)
                        )
                    for node in self.data.node]
        self.data['boundname'], self._data['boundname'] = [bdnmes]*2





    def get_data(self, istep=None, node=None, layer=None, i=None, j=None, boundname=None, force=False , as_mask=False):
        """
        Function to select/subset pumping data.

        Parameters:
        ----------
        istep (int, optional) : required timestep id(s).
                                If None, all timesteps wil be considered.
                                Default is None.
        node (int, optional) :  cell id(s).
                                Must be 0 < node < nnodes.
                                If None, all cells will be considered.
                                Default is None.
        layer (int, optional) : required layer id(s).
                                If None, all layers will be considered.
                                Default is None.
        i (int, optional) : required row id(s).
                            If None, all rows will be considered.
                            Default is None.
        j (int, optional) : required column id(s).
                            If None, all columns will be considered.
                            Default is None.
        boundname (str, optional) : required well name(s).
                                    If None, all boundnames will be considered.
                                    Default is None.
        force (bool, optional) : force getting pumping data for all required timesteps
                                 even if there are not provided explicitly in the pastp file.
                                 For a not provided required time step the nearest previous
                                 istep (npi) containing pumping data will be considered.
                                 Note: can be slow if the model contains a lot of timesteps.
        as_mask (bool) : returning data as boolean index.
                         Default is False.

        Returns:
        --------
        df (np.recarray) : subset DataFrame
        Note: if all arguments are set to None,
              all data is returned.

        Examples:
        --------
        df1 = mp.get_data(istep=[3,6,9,14], boundname = ['p1','p2'])
        df2 = mp.get_data(layer=2, i=33, j=18)

        """
        # ---- Get columns to perform queries
        col_query = self.data.drop('value', axis=1).columns

        # ---- Build query (format: q = 'column_0 in [value_0,..] & ...'')
        q = ' & '.join(
                ["{} in {}".format(k, list(self.data[k].unique())) 
                    if v is None else "{} in {}".format(k, list(marthe_utils.make_iterable(v)))
                    for k, v 
                    in zip(col_query, [istep,node,layer,i,j,boundname])
                    ]
                        )
        
        # ---- Force all provided isteps (slow)
        if force:
            # -- Subset (without timestep)
            dfs = []
            df_ss = self.data.query(q[q.index('node'):])
            for istep in range(self.mm.nstep):
                df = df_ss.loc[df_ss.istep == istep]
                # -- If istep not provided in pastp file
                if df.empty:
                    nip = df_ss.loc[df_ss.istep < istep, 'istep'].max()   # nip = nearest previous istep
                    np_df = df_ss[df_ss.istep == nip]
                    np_df['istep'] = istep
                    dfs.append(np_df)
                else:
                    dfs.append(df)
            # -- Finally concatenate all forced DataFrames
            df = pd.concat(dfs, ignore_index=True)
            # -- Set mask to None for fording process
            if as_mask:
                warn_msg = 'Getting data process with `as_mask = True` will not return any usable `mask`.'
                warnings.warn(warn_msg)
            mask = None

        # ---- Subset pastp provided isteps only (fast)
        else:
            # -- Get index of required values
            idx = self.data.query(q).index
            # -- Get data boolean mask
            mask = self.data.index.isin(idx)
            # -- Get subset data
            df = self.data.loc[idx]

        # ---- Return as required
        if as_mask:
            return mask
        # ---- Return subset data
        else:
            return df






    def set_data_from_parfile(self, parfile, keys, value_col, btrans):
        """
        """
        # -- Get all data
        df = self._data.copy(deep=True)
        # -- Get kmi and transformed values
        kmi, bvalues = pest_utils.parse_mlp_parfile(parfile, keys, value_col, btrans)
        # -- Convert to MultiIndex Dataframe
        mi_df = df.set_index(keys)
        # -- Set values and back to single index
        mi_df.loc[kmi, value_col] = bvalues.values
        data = mi_df.reset_index()
        # -- Set data inplace
        self.data, self._data = data[self.vars], data[self._vars]




    def set_data(self, value, istep=None, node=None, layer=None, i=None, j=None, boundname=None):
        """
        Function to set pumping data inplace.

        Parameters:
        ----------
        value (int/float) : pumping rate value to set.
        node (int, optional) :  cell id(s).
                                Must be 0 < node < nnodes.
                                If None, all cells will be considered.
        istep (int, optional) : number(s) of timestep required.
                                If None, all timesteps are considered.
                                Default is None.
        layer (int, optional) : number(s) of layer required.
                                If None, all layers are considered.
                                Default is None.
        i (int, optional) : number(s) of line required.
                            If None, all lines are considered.
                            Default is None.
        j (int, optional) : number(s) of column required.
                            If None, all columns are considered.
                            Default is None.
        boundname (str, optional) : name(s) required pumping point. 
                                    If None, all boundnames are considered.
                                    Default is None.

        Returns:
        --------
        Set pumping rate value inplace.

        Examples:
        --------
        mp.set_data(value = -44,67, istep=0, boundname = ['p1','p2'])
        mp.set_data(value = -189,4, layer=2, i=33, j=18)

        """
        # ---- Convert existing meta data (recarray) in DataFrame
        df = self._data.copy(deep=True)
        # ---- Get boolean mask of required data
        mask = self.get_data(istep, node, layer, i, j, boundname, as_mask=True)
        # ---- Change values in both data and metadata
        df.loc[mask, 'value'] = value
        # ---- Replace previous data
        self._data, self.data = df[self._vars], df[self.vars]




    def get_boundnames(self, istep=None, layer=None, i=None, j=None):
        """
        Function to get boundname on subset data.

        Parameters:
        ----------

        istep (int, optional) : number(s) of timestep required.
                                If None, all timesteps are considered.
                                Default is None.
        layer (int, optional) : number(s) of layer required.
                                If None, all layers are considered.
                                Default is None.
        i (int, optional) : number(s) of line required.
                            If None, all lines are considered.
                            Default is None.
        j (int, optional) : number(s) of column required.
                            If None, all columns are considered.
                            Default is None.

        Returns:
        --------
        boundnames (list) : available boundnames of selected data.

        Examples:
        --------
        steady_bdnmes = mp.get_boundnames(istep=0)
        bdnmes_l2 = mp.get_boundnames(layer=2)

        """
        # ---- Get boolean mask of wanted data
        mask = self.get_data(istep=istep, layer=layer, i=i, j=j, as_mask=True)
        # --- Extract boundname on subset data
        boundnames = self.data.loc[mask, 'boundname'].unique().tolist()
        # ---- Return boundnames as list of string
        return boundnames



    def switch_boundnames(self, switch_dic):
        """
        Function to change boundname of a pumping point by another.

        Parameters:
        ----------
        switch_dic (dict) : boundnames to switch
                            Format : {bdnme_source: bdnme_target, ...}

        Returns:
        --------
        Replace new boundname inplace.

        Examples:
        --------
        mp.switch_boundnames(switch_dic = {'B1951752/F1': 'F1'})
        """
        # ---- Use .replace method on boundname column
        self._data['boundname'].replace(switch_dic, inplace=True)
        self.data['boundname'].replace(switch_dic, inplace=True)




    def split_qtype(self, qtype=None):
        """
        Function to split pumping data into pandas Dataframe(s)
        according to the provided qtype ('mail', 'record' ,'listm')

        Parameters:
        ----------
        qtype (str/iterable) : type of pumping data to return
                               Can be (list) 'mail', 'record',
                               'listm' or None.
                               If None, all qtypes are provided Dataframe.
                               Default is None.

        Returns:
        --------
        [mail_df, record_df, listm_df] (list) : if qtype is None
        mail_df or record_df or listm_df (Dataframe) : is qtype is provided

        Examples:
        --------
        mail_df = mp.split_qtype('mail')[0]

        """
        # --- Manage required qtypes
        if qtype is None:
            qtypes = self.qtypes
        elif marthe_utils.isiterable(qtype):
            qtypes = qtype
        elif isinstance(qtype, str):
            qtypes = [qtype]

        # ---- group data by qtype
        gb = self._data.groupby('qtype')

        # ---- Split data by qtype
        dfs = [gb.get_group(qt) 
               if qt in self._data['qtype'].unique()
               else pd.DataFrame() 
               for qt in qtypes]

        # ---- Return list of (required) DataFrames
        return dfs





    def _write_mail(self):
        """
        Function to write pumping data (as 'mail' qtype) inplace.

        Parameters:
        ----------
        self (MarthePump) : instance.

        Returns:
        --------
        Write 'mail' pumping data in 
        parent model .pastp file.

        Examples:
        --------
        mm.prop['aqpump']._write_mail()

        """
        # ---- Fetch 'mail' qtype DataFrame
        mail_df = self.split_qtype('mail')[0]

        # ---- Convert back to 1-based
        mail_df[['layer', 'i', 'j']] = mail_df[['layer', 'i', 'j']].add(1)

        # ---- Define mode tag 
        mode_tag = '/DEBIT/' if self.mode == 'aquifer' else '/Q_EXTER_RIVI/'

        # ---- Set usefull regex
        sp = r'\s*'
        re_num = r"[-+]?\d*\.?\d+|\d+"
        re_istep = r"\*{3}\s*Le pas|Début"

        # ---- Fetch .pastp file content by lines
        with open(self.pastp_file, 'r') as f:
            lines = f.readlines()

        # ---- Initialize lines list and timestep counter
        new_lines = []
        istep = -1
        # ---- Iterate over lines
        for line in lines:
            if not re.search(re_istep, line) is None:
                # ---- Update istep
                istep += 1
                # ---- Get available value to replace
                df = mail_df.loc[mail_df['istep'] == istep]
                # ---- Update replace dictionary for this istep
                repl_dic = {}
                for i,row in df.iterrows():
                    c,l,p,v = row[['j','i','layer','value']].astype(str)
                    match = r''.join(['C=', sp, c, 'L=', sp, l, 
                                     'P=', sp, p, 'V=', sp, re_num])
                    repl = 'C={:>7}L={:>7}P={:>7}V={:>10}'.format(c,l,p,v)
                    repl_dic[match] = repl
                # ---- stock as new line
                new_lines.append(line)
            elif mode_tag in line:
                new_line = line
                # ---- Replace matches by new value
                for re_match, repl in repl_dic.items():
                    m = re_match.split('V=')[0]
                    if bool(re.search(m, line)):
                        new_line = re.sub(re_match, repl, line)
                # ---- Append (modified) line
                new_lines.append(new_line)
            else:
                # ---- Append line
                new_lines.append(line)

        # ---- Write all data
        with open(self.pastp_file, 'w') as f:
            f.write(''.join(new_lines))



    def _write_record(self):
        """
        Function to write pumping data (as 'record' qtype) inplace.

        Parameters:
        ----------
        self (MarthePump) : instance.

        Returns:
        --------
        Write 'record' pumping data in parent
        model .pastp file (steady-state data)
        and external file (transient data).

        Examples:
        --------
        mm.prop['aqpump']._write_record()

        """
        # ---- Fetch 'record' qtype DataFrame
        rec_df = self.split_qtype('record')[0]

        # ---- Convert back to 1-based
        rec_df[['layer', 'i', 'j']] = rec_df[['layer', 'i', 'j']].add(1)

        # ---- Set usefull regex
        re_block = r";\s*\*{3}\s*\n(.*?)/\*{5}"
        re_num = r"[-+]?\d*\.?\d+|\d+"
        re_jikv = r"C=\s*({})L=\s*({})P=\s*({})V=\s*({});".format(*[re_num]*4)

        # ---- Define mode tag
        mode_tag = '/DEBIT/' if self.mode == 'aquifer' else '/Q_EXTER_RIVI/'
        # ---- Extract pastp steady-state data (first data block)
        with open(self.pastp_file, 'r') as f:
            steady_block = re.findall(re_block, f.read(), re.DOTALL)[0]

        # ---- Rewrite steady-stade data in pastp file
        for line in steady_block.splitlines(True):
            if all(s in line for s in [mode_tag + 'MAIL', 'File=']):
                # ---- Fetch localisation infos
                c,l,p,v = map(ast.literal_eval, re.findall(re_jikv, line)[0])
                # ---- Query record DataFrame
                q = f'i == {l} & j == {c} & layer == {p} & istep == 0'
                new_v = rec_df.query(q)['value'].values[0]
                # ---- Change value
                new_line = re.sub(r'V=\s*{};'.format(re_num),
                                  'V={:>10}'.format(new_v),
                                  line)
                # ---- Change in file
                marthe_utils.replace_text_in_file(self.pastp_file, line, new_line)

        # ---- Rewrite transient data in external file       
        for qfilename in rec_df['qfilename'].unique():
            # ---- Read external file 
            df = pd.read_csv(qfilename,  delim_whitespace=True)
            # ---- Set new values
            rec_df_ss = rec_df.query(f"istep != 0 & qfilename == '{qfilename}'")
            for qcol, gb in rec_df_ss.groupby('qcol'):
                df.iloc[:,int(qcol)] = gb['value'].values
            # ---- Rewrite external file
            df.to_csv(qfilename, sep = '\t',header = True,index = False)




    def _write_listm(self):
        """
        Function to write pumping data (as 'listm' qtype) inplace.

        Parameters:
        ----------
        self (MarthePump) : instance.

        Returns:
        --------
        Write 'listm' pumping data in 
        external file.

        Examples:
        --------
        mm.prop['aqpump']._write_listm()

        """
        # ---- Fetch 'listm' qtype DataFrame
        listm_df = self.split_qtype('listm')[0]

        # ---- Convert back to 1-based
        listm_df[['layer', 'i', 'j']] = listm_df[['layer', 'i', 'j']].add(1)

        # ---- Write (modified) data
        for qfilename, data in listm_df.groupby('qfilename'):
            df = pd.read_csv(qfilename, header=None, delim_whitespace=True)
            for qcol, gb in data.groupby('qcol'):
                df.iloc[:,int(qcol)] = gb['value'].values
            df.to_csv(qfilename, sep='\t', header=False, index=False)




    def write_data(self):
        """
        Function to write pumping data inplace.
        (All qtypes are considered).

        Parameters:
        ----------
        self : MarthePump instance

        Returns:
        --------
        Write data inplace.
        (record files, listm files, pastp file)

        Examples:
        --------
        mp.set_data(value = 3, istep=[3,5])
        mp.write_data()

        """
        # ---- Split data according to qtype
        mail_df, record_df, listm_df = self.split_qtype()

        # ---- Write single cell / single pumping condition (mail)
        if not mail_df.empty:
            self._write_mail()

        # ---- Write single cell / multiple pumping condition (record)
        if not record_df.empty:
            self._write_record()

        # ---- Write multiple cell / single condition (listm)
        if not listm_df.empty:
            self._write_listm()


    def __str__(self):
        """
        Internal string method.
        """
        return 'MarthePump'
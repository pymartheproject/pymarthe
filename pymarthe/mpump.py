"""
Contains the MarthePump class (Subclass of MartheModel)
for handling pumping conditions by locations.
"""

import os
import numpy as np
import pandas as pd
import re, ast
from .utils import marthe_utils


encoding = 'latin-1'


class MarthePump():
    """
    Class for handling Marthe pumping data.

    """

    def __init__(self, mm, pastp_file = None,  mode = 'aquifer'):
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
        self.vars = ['istep', 'layer', 'i', 'j', 'value', 'boundname']
        self._vars = self.vars + ['qfilename', 'qtype', 'qrow', 'qcol']
        self.qtypes = ['mail', 'record', 'listm']

        # ---- Read pumping data (and metadata) from .pastp file according to pumping type (DataFrame)
        self._extract_data(mode)




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
            self.data, self._data = marthe_utils.extract_pastp_pumping(self.pastp_file, mode)
            # ---- Convert xy columns to ij
            if all(loc in self.data.columns for loc in list('xy')):
                # -- Print message to inform about convertion
                print('Converting xy pumping data into row(s), column(s) ...')
                # -- Fetch data as DataFrame
                _df = self._data.copy(deep=True)
                # -- Convert xy to ij (could take a while)
                _df['i'], _df['j'] = self.mm.get_ij(_df['x'], _df['y'])
                _df['boundname'] =_df['i'].astype(str) + '_' + _df['j'].astype(str)
                self.data, self._data = _df[self.vars], _df[self._vars]
        # ---- Manage 'rivpump'
        if self.mode == 'river':
            # ---- Convert aff/trc data in column, line, plan (layer) format in .pastp file
            marthe_utils.convert_at2clp_pastp(self.pastp_file, mm = self.mm)
            self.data, self._data = marthe_utils.extract_pastp_pumping(self.pastp_file, mode)




    def _make_iterable(self, var, var_name):
        """
        Function to convert input arguments
        of a function to iterable.

        Parameters:
        ----------
        var (object) : variable.
        var_name (str) : variable name.

        Returns:
        --------
        it (ietrable) : iterable.

        Examples:
        --------
        istep = 3
        mp._make_variable(istep, 'istep')
        """
        # ---- Manage var == None case
        if var is None:
            it = np.unique(self.data[var_name])
        # ---- Manage var != None case
        else: 
            it = marthe_utils.make_iterable(var)
        # ---- Return iterable
        return it




    def get_data(self, istep=None, layer=None, i=None, j=None, boundname=None, as_mask=False):
        """
        Function to select/subset data.

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
        boundname (str, optional) : name(s) required pumping point. 
                                    If None, all boundnames are considered.
                                    Default is None.
        as_mask (bool) : returning data as boolean index.
                         Default is False.

        Returns:
        --------
        rec (np.recarray) : selected data as recarray.
        Note: if all arguments are set to None,
              all data is returned.

        Examples:
        --------
        rec1 = mp.get_data(istep=[3,6,9,14], boundname = ['p1','p2'])
        rec2 = mp.get_data(layer=2, i=33, j=18)

        """

        # ---- Transform records to Dataframe for query purpose
        df = self.data.copy(deep=True)

        # ---- Get columns to perform queries
        col_query = df.drop('value', axis=1).columns

        # ---- Make all arguments iterable
        its = [self._make_iterable(var, var_name) 
               for var, var_name
               in zip([istep,layer,i,j,boundname], col_query)]

        # ---- Get query mask
        conditions = [df[col].isin(it).values for col,it in zip(col_query, its)]
        mask = np.logical_and.reduce(conditions)

        # ---- Return query as mask if required
        if as_mask:
            return mask
        # ---- Return data as recarray
        else:
            return df.loc[mask]





    def set_data(self, value, istep=None, layer=None, i=None, j=None, boundname=None):
        """
        Function to set pumping data inplace.

        Parameters:
        ----------
        value (int/float) : pumping rate value to set.
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
        mask = self.get_data(istep, layer, i, j, boundname, as_mask=True)
        # ---- change value 
        df.loc[mask, 'value'] = value
        # ---- replace previous data
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
        df = self.data.copy(deep=True)
        boundnames = df.loc[mask, 'boundname'].unique().tolist()
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
        # ---- Convert existing meta data (recarray) in DataFrame
        df = self._data.copy(deep=True)
        # ---- Use .replace method on boundname column
        df['boundname'] = df['boundname'].replace(switch_dic)
        # ---- Set new data
        self._data, self.data = df[self._vars], df[self.vars]



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

        # ---- Set usefull regex
        re_block = r";\s*\*{3}\s*\n(.*?)/\*{5}"
        re_num = r"[-+]?\d*\.?\d+|\d+"
        re_jikv = r"C=\s*({})L=\s*({})P=\s*({})V=\s*({});".format(*[re_num]*4)
        sp = r'\s*'

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
                new_line = re.sub('V=' + sp + re_num,
                                  'V={:>10}'.format(new_v),
                                  line)
                # ---- Change in file
                marthe_utils.replace_text_in_file(mp.pastp_file, line, new_line)

        # ---- Rewrite transient data in external file       
        for qfilename in rec_df['qfilename'].unique():
            # ---- Read external file 
            df = pd.read_csv(qfilename, dtype='f8',  delim_whitespace=True)
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

        # ---- Write (modified) data
        for qfilename, df in listm_df.groupby('qfilename'):
            df.to_csv(qfilename,
                      sep = '\t',
                      header = False,
                      index = False,
                      columns = ['value','j', 'i','layer'])




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
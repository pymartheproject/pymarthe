"""
Contains the MarthePump class (Subclass of MartheModel)
for handling pumping conditions by locations

"""

import os
import numpy as np
import pandas as pd
import re
from .utils import marthe_utils


encoding = 'latin-1'


class MarthePump():
    """
    Class for handling pumping data

    """


    def __init__(self, mm, pastp_file = None,  mode = 'aquifer'):
        """
        Instance generator of MarthePump class

        Parameters
        ----------
        mm : MarthePump instance
        pastp_file (str) : name (or full path) of the .pastp file
                           Default: name of the MartheModel parent class
        mode (str) : type of withdraw pumping
                     Can be 'aquifer' or 'river'
                     Default is 'aquifer'

        Examples
        --------
        mm = MartheModel(rma_file)
        mpump = MarthePump(mm)
        """
        # ---- Pointer to parent model
        self.mm = mm

        # ----Fetch pastp name if not precised
        if pastp_file is None:
            self.pastp_file = os.path.join(self.mm.mldir, self.mm.mlname + '.pastp')
        else:
            self.pastp_file = pastp_file

        # ---- Fet number of steps from pastp file
        self.nstep, self._content = marthe_utils.read_pastp(self.pastp_file)

        # ---- Read pumping data from .pastp file
        if mode == 'aquifer':
            # ---- Read aquifer pumping data
            self.pumping_data, self.qfilenames = marthe_utils.extract_pastp_pumping(self._content, mode = mode)
        if mode == 'river':
            # ---- Convert aff/trc data in column, line, plan (layer) format in pastp file
            marthe_utils.convert_at2clp_pastp(self.pastp_file, mm = self.mm)
            # ---- Reread pastp file after convertion
            self.nstep, self._content = marthe_utils.read_pastp(self.pastp_file)
            # ---- Read river pumping data
            self.pumping_data, self.qfilenames = marthe_utils.extract_pastp_pumping(self._content, mode = mode)




    def get_data(self, istep = None):
        """
        Function to fetch pumping data.
        Format : {istep_0 : arr_0, istep_1 : arr_1, ..., istep_N : arr_N}

        Parameters:
        ----------
        self : MarthePump instance
        istep (int) : required timestep (<= nstep)

        Returns:
        --------
        data (dict) : all pumping data (if istep is None)
        arr (array) : pumping data of a specific timestep 
                      (if istep is int <= nstep)

        Examples:
        --------
        mm = MartheModel(rma_file)
        mpump = mm.mpump.MarthePump(mm)
        rec8 = mpump.get_data(istep=8)
        """
        # ---- Check if istep argument is None
        if istep is None:
            # ---- Return all data
            return self.pumping_data
        else:
            # ---- Assert that istep is in [0:nstep]
            msg = f'ERROR : istep must be between 0 and {self.nstep-1}. Given {istep}' 
            assert 0 <= istep <= self.nstep, msg
            # ---- Return data for required timestep
            return self.pumping_data[istep]




    def set_data(self, pdata=None, istep=None, arr=None, v=None, c=None, l=None, p=None, boundname=None):
        """
        Function to set pumping data inplace

        Parameters:
        ----------
        self : MarthePump instance
        pdata (dict) : pumping data for all timestep
                       Format : {istep_0 : arr_0, istep_1 : arr_1, ..., istep_N : arr_N}
                       Default is None
        istep (int) : specific timestep
                      Must be 0 <= istep <= nstep
                      Default is None
        arr (np.ndarray) : pumping data for a given timestep
                           Format: array([(v1,c1,l1,p1), (v2,c2,l2,p2), ..., (vN,cN,lN,pN),
                                          dtype = ('V', '<f8'), ('C', '<i4'), ('L', '<i4'), ('P', '<i4')]])
                           Default is None
        v (numeric) : pumping value
                      Default is None
        c (int) : cell column
                  Default is None
        l (int) : cell line
                  Default is None
        p (int) : cell plan/layer
                  Default is None
        boundname (str) : name of the pumping cell
                          Default is None
        Returns:
        --------
        Set data inplace

        Examples:
        --------
        mm = MartheModel('mymarthemodel.rma')
        mm.add_pump()
        mm.mpump.set_data(pdata = mm.mpump.get_data())
        mm.mpump.set_data(istep = 3, arr = mm.mpump.get_data(3))
        mm.mpump.set_data(istep = 2, c = 18, l = 33, p = 1, v = -900.)
        mm.mpump.set_data(c = 18, l = 33, p = 1, v = -550.)
        """
        # ---- If all arguments are None
        if all(x is None for x in [pdata, istep, arr, v, c, l, p]):
            return print('No data provided.')

        # ---- If pdata is provided
        if isinstance(pdata, dict) & all(x is None for x in [istep, arr, v, c, l, p]):
            # -- Change the pumping data dict inplace 
            self.pumping_data = pdata

        # ---- If istep and arr is provided
        if all(isinstance(v,t) for v,t in zip([istep,arr], [int,np.ndarray])) & all(x is None for x in [pdata, v, c, l, p]):
            # -- Change the pumping data array for a given timestep
            self.pumping_data[istep] = arr

        # ----If istep, istep, c, l, p are provided
        if all(isinstance(v,int) for v in [istep, c, l, p]) & (v is not None) & all(x is None for x in [pdata, arr]):
            # -- Change the pumping data for a unique cell for a given timestep
            locc = self.pumping_data[istep]['c'] == c
            locl = self.pumping_data[istep]['l'] == l
            locp = self.pumping_data[istep]['p'] == p
            boundname = self.pumping_data[istep][locc & locl & locp]['boundname'][0]
            self.pumping_data[istep][locc & locl & locp] = (v,c,l,p,boundname)

        # ----If c, l, p are provided
        if all(isinstance(v,int) for v in [c, l, p]) & all(x is None for x in [pdata, arr, istep]):
            # -- Change the pumping data for a unique cell for a all timestep
            for istep in range(self.nstep):
                locc = self.pumping_data[istep]['c'] == c
                locl = self.pumping_data[istep]['l'] == l
                locp = self.pumping_data[istep]['p'] == p
                v = self.pumping_data[istep][locc & locl & locp]['v']
                if boundname is None:
                    boundname = self.pumping_data[istep][locc & locl & locp]['boundname'][0]
                self.pumping_data[istep][locc & locl & locp] = (v,c,l,p,boundname)

        # ----If boundname, v are provided (istep optional)
        if isinstance(boundname,str) & (v is not None) & all(x is None for x in [pdata, arr,c,l,p]):
            # -- Change the pumping data for a unique cell for a all timestep
            if isinstance(istep,int):
                locbnme = self.pumping_data[istep]['boundname'] == boundname
                c,l,p = self.pumping_data[istep][locbnme][0][list('clp')]
                self.pumping_data[istep][locbnme] = (v,c,l,p,boundname)
            else:
                for istep in range(self.nstep):
                    locbnme = self.pumping_data[istep]['boundname'] == boundname
                    c,l,p = self.pumping_data[istep][locbnme][0][list('clp')]
                    self.pumping_data[istep][locbnme] = (v,c,l,p,boundname)




    def get_boundnames(self, istep=None):
        """
        -----------
        Description:
        -----------
        Fetch boundnames of pumping cell
        
        Parameters: 
        -----------
        self : MarthePump instance
        istep (int) : required timestep (<= nstep)

        Returns:
        -----------
        boundnames (list) : pumping cell boundnames as list 

        Example
        -----------
        mm = MartheModel(rma_file)
        mm.add_pump()
        all_boundnames = mm.mpump.get_boundnames()
        """
        # ---- Get boundnames of a specific timestep
        if not istep is None:
            # ---- Assert that istep is in [0:nstep]
            msg = f'ERROR : istep must be between 0 and {self.nstep -1}. Given {istep}' 
            assert 0 <= istep <= self.nstep, msg
            # ---- Extract boundname for istep
            boundnames = np.unique(self.get_data(istep)['boundname']).tolist()
        # ---- Get boundnames over all timesteps
        else:
            df = pd.concat(self.split_qtype())
            boundnames = df['boundname'].unique().tolist()
        # ---- Return boundnames as list of string
        return boundnames




    def switch_boundnames(self, switch_dic):
        """
        Function to change boundname of a pumping cell by another

        Parameters:
        ----------
        self : MarthePump instance
        switch_dic (dict) : boundnames to switch
                            Format : {bdnme_source: bdnme_target, ...}
        Returns:
        --------
        Set data inplace

        Examples:
        --------
        mm = MartheModel(rma_file)
        mpump = mm.mpump.MarthePump(mm)
        mpump.switch_boundnames(switch_dic = {'B1951752/F1': 'F1'})
        """
        for bdnme_src, bdnme_tar in switch_dic.items():
            # ---- Change source boundname by new one for a unique cell for a all timestep
            for istep in range(self.nstep):
                locbnme = self.pumping_data[istep]['boundname'] == bdnme_src
                v,c,l,p = self.pumping_data[istep][locbnme][0][list('vclp')]
                self.pumping_data[istep][locbnme] = (v,c,l,p,bdnme_tar)



    def get_qfilenames(self, istep = None):
        """
        Function to fetch pumping data filenames.
        Format : {istep_0 : qfilename_0, ..., istep_N : qfilename_N}

        Parameters:
        ----------
        self : MarthePump instance
        istep (int) : required timestep (<= nstep)

        Returns:
        --------
        qfilenames (dict) : all pumping data filenames (if istep is None)
        qfilename  (str) : pumping data filename of a specific timestep 
                           (if istep is int <= nstep)

        Examples:
        --------
        mm = MartheModel(rma_file)
        mpump = mm.mpump.MarthePump(mm)
        rec8 = mpump.get_qfilenames(istep=8)
        """
        if istep is None:
            return self.qfilenames
        else:
            return self.qfilenames[istep]




    def split_qtype(self, qtype = None):
        """
        Function to split pumping data into pandas Dataframe
        according to the provided qtype ('Mail', 'Record' ,'Listm')

        Parameters:
        ----------
        self : MarthePump instance
        qtype (str) : type of pumping data to return
                      Can be 'mail', 'record', 'listm' or None
                      If None is provided, return all qtype Dataframe
                      Default is None

        Returns:
        --------
        [mail_df, record_df, listm_df] (list) : if qtype is None
        mail_df or record_df or listm_df (Dataframe) : is qtype is provided

        Examples:
        --------
        mm = MartheModel(rma_file)
        mm.add_pump()
        mail_df, record_df, listm_df = mm.pump.split_qtype()
        """
        # ---- Initialization
        qtypes = ['Mail','Record','Listm']
        qtype_dfs = dict.fromkeys([s.lower() for s in qtypes])
        flat_data = []
        istep = -1

        # ---- Iterate over pumping and filenames arrays (by timestep)
        for file_arr, pump_arr in zip(self.get_qfilenames().values(), self.get_data().values()):
            # ---- Get istep
            istep += 1
            # ---- Iterate over each pumping condition in timestep
            for file, pump in zip(file_arr, pump_arr):
                # ---- Get basic pumping information
                v, c, l, p, bdnme = pump
                # ---- Manage 'Mail' 
                if file is None:
                    qt = 'Mail'
                    qfilename = file
                    j, i = [None]*2
                else:
                    # ---- Manage 'Record' 
                    if 'Record' in file:
                        qt = 'Record'
                        qfilename, tag = file.split('&')
                        j,i = map(int, re.findall(r'(\d+)', tag))
                    # ---- Manage 'Listm' 
                    if 'Listm' in file:
                        qt = 'Listm'
                        qfilename, tag = file.split('&')
                        j,i = [None] + list(map(int, re.findall(r'(\d+)', tag)))
                # ---- Add flat data
                flat_data.append([qt, istep, qfilename, i, j, v, c, l, p, bdnme])

        # ---- Convert flat data into DataFrame (fixing dtypes)
        columns = ['qtype', 'istep', 'qfilename'] + list('ijvclp') + ['boundname']
        df = pd.DataFrame(flat_data, columns= columns)

        # ---- Split all data according to qtype
        for qt in qtypes:
            qtype_dfs[qt.lower()] = df.query(f"qtype == '{qt}'")

        # ---- Return
        if qtype is None:
            return list(qtype_dfs.values())
        else:
            return qtype_dfs[qtype.lower()]






    def _write_mail(self, df):
        """
        Function to write pumping data inplace

        Parameters:
        ----------
        self : MarthePump instance
        df (DataFrame) : Dataframe to write
                         Obtained from .split_qtype('mail')

        Returns:
        --------
        Write data inplace

        Examples:
        --------
        mm = MartheModel(rma_file)
        mm.add_pump()
        mail_df = mm.pump.split_qtype('mail')
        mm.pump._write_record(mail_df)
        """
        # ---- Iterate over timestep
        for istep in df['istep'].unique():

            # ---- Subset Dataframe for specific timestep
            df_ss = df.loc[df['istep'] == istep, list('vclp')]
            
            # ---- Iterate over each cell (with pumping condition) in timestep
            for idx, row in df_ss.iterrows():
                # -- Fetch basic informations (value, column, row, layer)
                v, c, l, p = row
                slocs = [str(int(num)) for num in [c, l, p]]

                # -- Set regex of 'any whitespace'
                sp = r'(\s*)'
                # -- Set regex of column, row, layer matching
                re_loc = r''.join([loc + sp + str(val) for loc, val in zip(['C=','L=','P='],slocs)])
                # -- Set regex of pumping condition type (/'condition'/'condition type') matching
                re_type = r'/DEBIT/MAILLE'
                # -- Join cell and condition regex matching 
                re2match = sp + re_type + sp + re_loc
                # -- Extract matched line as string
                lmatch = [line.strip('\n') for line in self._content[istep] if re.match(re2match, line)][0]
                # -- Extract value section in matched string
                val = lmatch[lmatch.index('V='):lmatch.index(';')]
                # -- Set the current value store in MarthePump instance (/!\ Format)
                new_val =  'V=' + '{:>10}'.format(str(v))
                # -- Define the string to replace
                l2replace = lmatch.replace(val, new_val)
                # -- Actually replace value in .pastp file (inplace)
                marthe_utils.replace_text_in_file(self.pastp_file, lmatch, l2replace)
                # -- Update pastp content in current MarthePump instance by reading modified .pastp file
                self.nstep, self._content = marthe_utils.read_pastp(self.pastp_file)




    def _write_record(self, df):
        """
        Function to write pumping data inplace

        Parameters:
        ----------
        self : MarthePump instance
        df (DataFrame) : Dataframe to write
                         Obtained from .split_qtype('record')

        Returns:
        --------
        Write data inplace

        Examples:
        --------
        mm = MartheModel(rma_file)
        mm.add_pump()
        record_df = mm.pump.split_qtype('record')
        mm.pump._write_record(record_df)
        """
       # ---- Iterate over external files
        for qfilename in df['qfilename'].unique():
            # ---- Iterate over columns in file
            dfs_col = []
            for j in df['j'].unique():
                # ---- Subset by qfilename/columns and sort by row (i)
                condition = (df['qfilename'] == qfilename) & (df['j'] == j)
                df_ss = df[condition].sort_values('i').reset_index()
                # ---- Extract pumping record as single column dataframe
                header = df_ss['boundname'].unique()[0]
                df_col = df_ss[['v']].rename(columns = {'v':header})
                # ---- Add record
                dfs_col.append(df_col)
            # ---- Concatenate all records
            df_records = pd.concat(dfs_col, axis=1)
            # ---- Write recors
            df_records.to_csv(qfilename,
                              sep = '\t',
                              header = True,
                              index = False)




    def _write_listm(self, df):
        """
        Function to write pumping data inplace

        Parameters:
        ----------
        self : MarthePump instance
        df (DataFrame) : Dataframe to write
                         Obtained from .split_qtype('listm')

        Returns:
        --------
        Write data inplace

        Examples:
        --------
        mm = MartheModel(rma_file)
        mm.add_pump()
        listm_df = mm.pump.split_qtype('listm')
        mm.pump._write_listm(listm_df)
        """
        # ---- Iterate over external file
        for qfilename in df['qfilename'].unique():
            # ---- Subset by qfilename and sort by row (i) 
            df_ss = df[df['qfilename'] == qfilename].sort_values('i')
            # ---- Write DataFrame in file
            df_ss.to_csv(qfilename,
                         columns = list('vclp'),
                         sep='\t',
                         header=False,
                         index=False)





    def write_data(self):
        """
        Function to write pumping data inplace

        Parameters:
        ----------
        self : MarthePump instance

        Returns:
        --------
        Write data inplace (record files, listm files, pastp file)

        Examples:
        --------
        mm = MartheModel(rma_file)
        mm.add_pump()
        mm.write_data()
        """
        # ---- Split data according to qtype
        mail_df, record_df, listm_df = self.split_qtype()

        # ---- Write single cell / single pumping condition (mail)
        if not mail_df.empty:
            self._write_mail(mail_df)

        # ---- Write single cell / multiple pumping condition (record)
        if not record_df.empty:
            self._write_record(record_df)

        # ---- Write multiple cell / single condition (listm)
        if not listm_df.empty:
            self._write_listm(listm_df)






    
    def write_param(self, prefix = 'q', param_dir = '.', boundname = None):
        """
        Function to write pumping data as parameter file (pest purpose)

        Parameters:
        ----------
        self : MarthePump instance
        prefix (str) : prefix of parameter file and parameter names
                       Make sure to choose the shortest prefix possible
                       Default is 'q'
        param_dir (str) : folder path to write parameter(s)
                          Default is '.' (current directory)
        boundname (str) : pumping cell name to write
                          Default is None
                          If boundname is None, all boundname are taken into account
        Returns:
        --------
        Write parameter files inplace

        Examples:
        --------
        mm = MartheModel(rma_file)
        mm.add_pump()
        mm.write_param(prefix ='q', param_dir = 'qparams', boundname = 'F1')
        """
        # ---- Get number of digits of timesteps
        nd = len(str(self.nstep))
        # ---- Get pumping data information as large DataFrame
        qt_df = pd.concat(self.split_qtype())
        # ---- Subset by boundname if required
        df = qt_df if boundname is None else qt_df.loc[qt_df['boundname'] == boundname]
        # ---- Format istep string
        df['istep_tpl'] = df['istep'].astype(str).apply(lambda s: s.zfill(nd))
        # ---- Build parnme columns
        df['parnme'] = df[['boundname','istep_tpl']].apply(lambda s:'{0}_{1}'.format(prefix,'_'.join(s)), axis=1)
        # ---- Set headers for param file
        headers = ['v','c','l','p','istep','boundname','parnme']
        for bdnme in df['boundname'].unique():
            # ---- Subset by boundname
            bdnme_df = df.loc[df['boundname'] == bdnme]
            # ---- Write param file for boundname
            param_file = os.path.join(param_dir, f'{prefix}_{bdnme}.dat')
            bdnme_df[headers].to_csv(param_file, header = True, index = False, sep = '\t')



    def read_param(self, param_file):
        """
        Function to read parameter file

        Parameters:
        ----------
        self : MarthePump instance
        param_file (str) : path to the parameter file

        Returns:
        --------
        param_df (Dataframe) : parameter information

        Examples:
        --------
        mm = MartheModel(rma_file)
        mm.add_pump()
        param_df = mm.mpump.read_param('qparams/q_f2.dat')
        """
        # ---- Read all parameter files 
        param_df = pd.read_csv(param_file, header=0, sep ='\t')
        # ---- Return parameter DataFrame
        return param_df




    def set_param(self, param_files):
        """
        Function to set parameter file and push it to MarthePump pumping data

        Parameters:
        ----------
        self : MarthePump instance
        param_files (list) : path to the parameter files

        Returns:
        --------
        Set data in mpump.pumping_data attribut

        Examples:
        --------
        mm = MartheModel(rma_file)
        mm.add_pump()
        param_files = ['qparams/q_p1.dat','qparams/q_p2.dat']
        mm.mpump.set_param(param_files)
        """
        # ---- Get pumping parameters as DataFrame
        param_df = pd.concat([self.read_param(param_file) for param_file in param_files])
        # ---- Set array dtypes (mpump structure array)
        dt = [('v','f8'), ('c', 'i4'), ('l', 'i4'), ('p', 'i4'), ('boundname', '<U25')]
        # ---- Iterate over timesteps (subset by istep)
        for istep , df in param_df.groupby('istep'):
            # ---- Convert DataFramen (subset) to array
            raw_arr = df[['v','c','l','p','boundname']].to_numpy()
            # ---- Fix dtype
            arr = np.array(list(map(tuple, raw_arr)), dtype=dt)
            # ---- Set array in mpump
            self.set_data(istep=istep, arr=arr)




    def write_tpl(self, prefix = 'q', tpl_dir = '.', boundname = None):
        """
        Function to pest template file of pumping data (pest purpose)

        Parameters:
        ----------
        self : MarthePump instance
        prefix (str) : prefix of tpl file and parameter names
                       Make sure to choose the shortest prefix possible
                       Default is 'q'
        tpl_dir (str) : folder path to write template file
                        Default is '.' (current directory)
        boundname (str) : pumping cell name 
                          Default is None
                          If boundname is None, all boundname are taken into account
        Returns:
        --------
        Write parameter files inplace

        Examples:
        --------
        mm = MartheModel(rma_file)
        mm.add_pump()
        mm.write_tpl(prefix ='q', tpl_dir = 'tpl', boundname = 'P1')
        """
        # ---- Get number of digits of timesteps
        nd = len(str(self.nstep))
        # ---- Get pumping data information as large DataFrame
        qt_df = pd.concat(self.split_qtype())
        # ---- Subset by boundname if required
        df = qt_df if boundname is None else qt_df.loc[qt_df['boundname'] == boundname]
        # ---- Format istep string
        df['istep_tpl'] = df['istep'].astype(str).apply(lambda s: s.zfill(nd))
        # ---- Build parnme columns
        df['parnme'] = df[['boundname','istep_tpl']].apply(lambda s:'{0}_{1}'.format(prefix,'_'.join(s)), axis=1)
        # ---- Build parnme columns for tpl pupose
        df['v'] = df['parnme'].apply(lambda s: f'~ {s} ~')
        # ---- Set headers for param file
        headers = ['v','c','l','p','istep','boundname','parnme']
        for bdnme in df['boundname'].unique():
            # ---- Subset by boundname
            bdnme_df = df.loc[df['boundname'] == bdnme]
            # ---- Get param file for boundname
            param_file = os.path.join(tpl_dir, f'{prefix}_{bdnme}.tpl')
            # ---- Open template file
            with open(param_file,'w',encoding=encoding) as f:
                # ---- Write template file tag (and delimiter)
                f.write('ptf ~\n')
                # ---- Write infos
                bdnme_df[headers].to_csv(f, header = True, index = False, sep = '\t')


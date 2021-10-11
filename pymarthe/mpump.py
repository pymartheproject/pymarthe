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

    Parameters
    ----------
    mm : MarthePump instance
    pastp_file (str) : name (or full path) of the .pastp file
                       Default: name of the MartheModel parent class

    Examples
    --------
    mm = MartheModel(rma_file)
    mpump = MarthePump(mm)
    """

    def __init__(self, mm,  pastp_file = None):

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
        self.pumping_data, self.qfilenames = marthe_utils.extract_pastp_pumping(self._content)



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
            msg = f'ERROR : istep must be between 0 and {self.nstep}. Given {istep}' 
            assert 0 <= istep <= self.nstep, msg
            # ---- Return data for required timestep
            return self.pumping_data[istep]




    def set_data(self, pdata=None, istep=None, arr=None, v=None, c=None, l=None, p=None):
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

        # ----If istep, v, c, l, p are provided
        if all(isinstance(v,int) for v in [istep, c, l, p]) & (v is not None) & all(x is None for x in [pdata, arr]):
            # -- Change the pumping data for a unique cell for a given timestep
            locc = self.pumping_data[istep]['C'] == c
            locl = self.pumping_data[istep]['L'] == l
            locp = self.pumping_data[istep]['P'] == p
            self.pumping_data[istep][locc & locl & locp] = (v,c,l,p)

            # ----If v, c, l, p are provided
        if all(isinstance(v,int) for v in [c, l, p]) & (v is not None) & all(x is None for x in [pdata, arr, istep]):
            # -- Change the pumping data for a unique cell for a given timestep
            for istep in range(self.nstep):
                locc = self.pumping_data[istep]['C'] == c
                locl = self.pumping_data[istep]['L'] == l
                locp = self.pumping_data[istep]['P'] == p
                self.pumping_data[istep][locc & locl & locp] = (v,c,l,p)




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
                v, c, l, p = pump
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
                flat_data.append([qt, istep, qfilename, i, j, v, c, l, p])

        # ---- Convert flat data into DataFrame (fixing dtypes)
        columns = ['qtype', 'istep', 'qfilename'] + list('ijvclp')
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
                header = 'X{}_Y{}'.format(*df_ss.iloc[0,7:9])
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







    

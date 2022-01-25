
"""
Contains geospatial export utils
(not much dependencies required)

"""

import numpy as np
import shutil
import shapefile

srefhttp = "https://spatialreference.org"


def shp2points(shpname, stack=True):
    """
    Extract xy-coordinates from a point shapefile.

    Parameters
    ----------
    shpname (str) : path to the points shapefile.
    stack (bool) : return coordinates type.
                   If True: [[x0,y0],[x1,y1],.. , [xN,yN]]
                   If False: np.array([x0,x1,.. xN], [y0,y1,.. yN])
                   Default is True

    """
    # ---- Initialize points shapefile reader
    r = shapefile.Reader(shpname)
    # ---- Assert that the shapefile contains points
    err_msg = 'ERROR : `shpname` must be shapefile of points. ' \
              f'Given shapefile type: `{r.shapeTypeName}`.'
    assert r.shapeTypeName == 'POINT', err_msg
    # ---- Extract points from shapefile
    points = []
    for shape in r.iterShapes():
        points.extend(shape.points)
    # ---- Return stack/unstack coordinates
    if stack:
        return points
    else:
        return np.column_stack(points)




def get_parts(xcc, ycc, dx, dy):
    """
    Return list of polygons parts from points
    considered as the centers of each polygons

    Parameters
    ----------
    xcc, ycc (float) : (xy)cellcenter coordinates
    dx, dy (float) : width, height of model cell

    """
    # ---- Fetch mesh grid 
    xx, yy = np.meshgrid(xcc, ycc)
    dxx, dyy = np.meshgrid(dx, dy)
    # ---- Transform to 1D arrays
    X, Y, DX, DY = list(map(np.ravel, [xx,yy,dxx,dyy]))
    # ---- Convert to list of points defining a polygon
    polygons = []
    for x,y,dx,dy in zip(X,Y,DX,DY):
        xl, xu = x - dx/2, x + dx/2
        yl, yu = y - dy/2, y + dy/2
        polygon = [[[xl,yl], [xl,yu], [xu,yu], [xu,yl], [xl,yl]]]
        polygons.append(polygon)
    # ---- Return list of polygons
    return polygons



def enforce_10ch_limit(names):
    """
    Enforce 10 character limit for fieldnames.
    Add suffix for duplicate names starting at 0.

    Parameters
    ----------
    names (list) : strings unformated

    Returns
    -------
    names : list of unique strings of len <= 10.
    """
    names = [n[:5] + n[-4:] + "_" if len(n) > 10 else n for n in names]
    dups = {x: names.count(x) for x in names}
    suffix = {n: list(range(cnt)) for n, cnt in dups.items() if cnt > 1}
    for i, n in enumerate(names):
        if dups[n] > 1:
            names[i] = n[:9] + str(suffix[n].pop(0))
    return names


def get_pyshp_field_info(dtypename):
    """Get pyshp dtype information for a given numpy dtype."""
    fields = {
        "int": ("N", 18, 0),
        "<i": ("N", 18, 0),
        "float": ("F", 20, 12),
        "<f": ("F", 20, 12),
        "bool": ("L", 1),
        "b1": ("L", 1),
        "str": ("C", 50),
        "object": ("C", 50),
    }
    k = [k for k in fields.keys() if k in dtypename.lower()]
    if len(k) == 1:
        return fields[k[0]]
    else:
        return fields["str"]


def get_pyshp_field_dtypes(code):
    """Returns a numpy dtype for a pyshp field type."""
    dtypes = {
        "N": int,
        "F": float,
        "L": bool,
        "C": object,
    }
    return dtypes.get(code, object)



def recarray2shp(recarray, geoms, shpname="recarray.shp", epsg=None, prj=None, **kwargs):
    """
    Write a numpy record array to a shapefile, using a corresponding
    list of geometries.
    Modify from librairy:
    https://github.com/modflowpy/flopy/blob/develop/flopy/export/shapefile_utils.py

    Parameters
    ----------
    recarray : np.recarray
        Numpy record array with attribute information that will go in the
        shapefile
    geoms : list of polygons 
        The number of geometries in geoms must equal the number of records in
        recarray.
    shpname : str
        Path for the output shapefile
    epsg : int
        EPSG code. See https://www.epsg-registry.org/ or spatialreference.org
    prj : str
        Existing projection file to be used with new shapefile.
    Notes
    -----
    Uses pyshp.
    epsg code requires an internet connection the first time to get the
    projection file text from spatialreference.org, but then stashes the text
    in the file epsgref.json (located in the user's data directory) for
    subsequent use. See flopy.reference for more details.
    """
    # ---- Check recarray/geoms respective length
    if len(recarray) != len(geoms):
        raise IndexError(
            "Number of geometries must equal the number of records!"
        )
    # ---- Check empty recarray
    if len(recarray) == 0:
        raise Exception("Recarray is empty")
    # ---- Initialize pyshp writer object
    w = shapefile.Writer(shpname, shapeType=shapefile.POLYGON)
    w.autoBalance = 1

    # ---- Write field for each name of recarray
    names = enforce_10ch_limit(recarray.dtype.names)
    for i, npdtype in enumerate(recarray.dtype.descr):
        key = names[i]
        if not isinstance(key, str):
            key = str(key)
        w.field(key, *get_pyshp_field_info(npdtype[1]))

    # ---- write the geometry and attributes for each record
    ralist = recarray.tolist()
    for i, r in enumerate(ralist):
        w.poly(geoms[i])
        w.record(*r)

    # ---- Close pyshp writer object
    w.close()
    # ---- Write projection file
    write_prj(shpname, epsg, prj)



def write_prj(shpname, epsg=None, prj=None, wkt_string=None):
    """
    Write a projection file (.proj).
    Figure which CRS option to use (prioritize args over grid reference)
    option to create prjfile from proj4 string without OGR or pyproj dependencies.
    """
    # ---- Get projection filename
    prjname = shpname.replace(".shp", ".prj")
    prjtxt = wkt_string
    # ---- Check epsg
    if epsg is not None:
        prjtxt = CRS.getprj(epsg)
    # ---- Copy a supplied prj file
    elif prj is not None:
        shutil.copy(prj, prjname)
    # ---- Print message if no proj info provided
    else:
        print(
            "No CRS information for writing a .prj file.\n"
            "Supply an epsg code or .prj file path to the "
            "model spatial reference or .export() method."
            "(writing .prj files from proj4 strings not supported)"
        )
    if prjtxt is not None:
        with open(prjname, "w") as output:
            output.write(prjtxt)


def get_url_text(url, error_msg=None):
    """
    Get text from a url.
    """
    from urllib.request import urlopen

    try:
        urlobj = urlopen(url)
        text = urlobj.read().decode()
        return text
    except:
        e = sys.exc_info()
        print(e)
        if error_msg is not None:
            print(error_msg)
        return



class CRS:
    """
    Container to parse and store coordinate reference system parameters,
    and translate between different formats.
    """

    def __init__(self, prj=None, esri_wkt=None, epsg=None):

        self.wktstr = None
        if prj is not None:
            with open(prj) as prj_input:
                self.wktstr = prj_input.read()
        elif esri_wkt is not None:
            self.wktstr = esri_wkt
        elif epsg is not None:
            wktstr = CRS.getprj(epsg)
            if wktstr is not None:
                self.wktstr = wktstr
        if self.wktstr is not None:
            self.parse_wkt()

    @property
    def crs(self):
        """
        Dict mapping crs attributes to proj4 parameters
        """
        proj = None
        if self.projcs is not None:
            # projection
            if "mercator" in self.projcs.lower():
                if (
                    "transvers" in self.projcs.lower()
                    or "tm" in self.projcs.lower()
                ):
                    proj = "tmerc"
                else:
                    proj = "merc"
            elif (
                "utm" in self.projcs.lower() and "zone" in self.projcs.lower()
            ):
                proj = "utm"
            elif "stateplane" in self.projcs.lower():
                proj = "lcc"
            elif "lambert" and "conformal" and "conic" in self.projcs.lower():
                proj = "lcc"
            elif "albers" in self.projcs.lower():
                proj = "aea"
        elif self.projcs is None and self.geogcs is not None:
            proj = "longlat"

        # datum
        datum = None
        if (
            "NAD" in self.datum.lower()
            or "north" in self.datum.lower()
            and "america" in self.datum.lower()
        ):
            datum = "nad"
            if "83" in self.datum.lower():
                datum += "83"
            elif "27" in self.datum.lower():
                datum += "27"
        elif "84" in self.datum.lower():
            datum = "wgs84"

        # ellipse
        ellps = None
        if "1866" in self.spheroid_name:
            ellps = "clrk66"
        elif "grs" in self.spheroid_name.lower():
            ellps = "grs80"
        elif "wgs" in self.spheroid_name.lower():
            ellps = "wgs84"

        return {
            "proj": proj,
            "datum": datum,
            "ellps": ellps,
            "a": self.semi_major_axis,
            "rf": self.inverse_flattening,
            "lat_0": self.latitude_of_origin,
            "lat_1": self.standard_parallel_1,
            "lat_2": self.standard_parallel_2,
            "lon_0": self.central_meridian,
            "k_0": self.scale_factor,
            "x_0": self.false_easting,
            "y_0": self.false_northing,
            "units": self.projcs_unit,
            "zone": self.utm_zone,
        }

    @property
    def grid_mapping_attribs(self):
        """
        Map parameters for CF Grid Mappings
        http://http://cfconventions.org/cf-conventions/cf-conventions.html,
        Appendix F: Grid Mappings
        """
        if self.wktstr is not None:
            sp = [
                p
                for p in [
                    self.standard_parallel_1,
                    self.standard_parallel_2,
                ]
                if p is not None
            ]
            sp = sp if len(sp) > 0 else None
            proj = self.crs["proj"]
            names = {
                "aea": "albers_conical_equal_area",
                "aeqd": "azimuthal_equidistant",
                "laea": "lambert_azimuthal_equal_area",
                "longlat": "latitude_longitude",
                "lcc": "lambert_conformal_conic",
                "merc": "mercator",
                "tmerc": "transverse_mercator",
                "utm": "transverse_mercator",
            }
            attribs = {
                "grid_mapping_name": names[proj],
                "semi_major_axis": self.crs["a"],
                "inverse_flattening": self.crs["rf"],
                "standard_parallel": sp,
                "longitude_of_central_meridian": self.crs["lon_0"],
                "latitude_of_projection_origin": self.crs["lat_0"],
                "scale_factor_at_projection_origin": self.crs["k_0"],
                "false_easting": self.crs["x_0"],
                "false_northing": self.crs["y_0"],
            }
            return {k: v for k, v in attribs.items() if v is not None}

    @property
    def proj4(self):
        """
        Not implemented yet
        """
        return None

    def parse_wkt(self):

        self.projcs = self._gettxt('PROJCS["', '"')
        self.utm_zone = None
        if self.projcs is not None and "utm" in self.projcs.lower():
            self.utm_zone = self.projcs[-3:].lower().strip("n").strip("s")
        self.geogcs = self._gettxt('GEOGCS["', '"')
        self.datum = self._gettxt('DATUM["', '"')
        tmp = self._getgcsparam("SPHEROID")
        self.spheroid_name = tmp.pop(0)
        self.semi_major_axis = tmp.pop(0)
        self.inverse_flattening = tmp.pop(0)
        self.primem = self._getgcsparam("PRIMEM")
        self.gcs_unit = self._getgcsparam("UNIT")
        self.projection = self._gettxt('PROJECTION["', '"')
        self.latitude_of_origin = self._getvalue("latitude_of_origin")
        self.central_meridian = self._getvalue("central_meridian")
        self.standard_parallel_1 = self._getvalue("standard_parallel_1")
        self.standard_parallel_2 = self._getvalue("standard_parallel_2")
        self.scale_factor = self._getvalue("scale_factor")
        self.false_easting = self._getvalue("false_easting")
        self.false_northing = self._getvalue("false_northing")
        self.projcs_unit = self._getprojcs_unit()

    def _gettxt(self, s1, s2):
        s = self.wktstr.lower()
        strt = s.find(s1.lower())
        if strt >= 0:  # -1 indicates not found
            strt += len(s1)
            end = s[strt:].find(s2.lower()) + strt
            return self.wktstr[strt:end]

    def _getvalue(self, k):
        s = self.wktstr.lower()
        strt = s.find(k.lower())
        if strt >= 0:
            strt += len(k)
            end = s[strt:].find("]") + strt
            try:
                return float(self.wktstr[strt:end].split(",")[1])
            except (
                IndexError,
                TypeError,
                ValueError,
                AttributeError,
            ):
                pass

    def _getgcsparam(self, txt):
        nvalues = 3 if txt.lower() == "spheroid" else 2
        tmp = self._gettxt(f'{txt}["', "]")
        if tmp is not None:
            tmp = tmp.replace('"', "").split(",")
            name = tmp[0:1]
            values = list(map(float, tmp[1:nvalues]))
            return name + values
        else:
            return [None] * nvalues

    def _getprojcs_unit(self):
        if self.projcs is not None:
            tmp = self.wktstr.lower().split('unit["')[-1]
            uname, ufactor = tmp.strip().strip("]").split('",')[0:2]
            ufactor = float(ufactor.split("]")[0].split()[0].split(",")[0])
            return uname, ufactor
        return None, None

    @staticmethod
    def getprj(epsg, addlocalreference=True, text="esriwkt"):
        """
        Gets projection file (.prj) text for given epsg code from
        spatialreference.org
        See: https://www.epsg-registry.org/
        Parameters
        ----------
        epsg : int
            epsg code for coordinate system
        addlocalreference : boolean
            adds the projection file text associated with epsg to a local
            database, epsgref.json, located in the user's data directory.
        Returns
        -------
        prj : str
            text for a projection (*.prj) file.
        """
        epsgfile = EpsgReference()
        wktstr = epsgfile.get(epsg)
        if wktstr is None:
            wktstr = CRS.get_spatialreference(epsg, text=text)
        if addlocalreference and wktstr is not None:
            epsgfile.add(epsg, wktstr)
        return wktstr

    @staticmethod
    def get_spatialreference(epsg, text="esriwkt"):
        """
        Gets text for given epsg code and text format from spatialreference.org
        Fetches the reference text using the url:
            https://spatialreference.org/ref/epsg/<epsg code>/<text>/
        See: https://www.epsg-registry.org/
        Parameters
        ----------
        epsg : int
            epsg code for coordinate system
        text : str
            string added to url
        Returns
        -------
        url : str
        """

        epsg_categories = (
            "epsg",
            "esri",
        )
        urls = []
        for cat in epsg_categories:
            url = f"{srefhttp}/ref/{cat}/{epsg}/{text}/"
            urls.append(url)
            result = get_url_text(url)
            if result is not None:
                break
        if result is not None:
            return result.replace("\n", "")
        elif result is None and text != "epsg":
            error_msg = (
                f"No internet connection or epsg code {epsg} not found at:\n"
            )
            for idx, url in enumerate(urls):
                error_msg += f"  {idx + 1:>2d}: {url}\n"
            print(error_msg)
        # epsg code not listed on spatialreference.org
        # may still work with pyproj
        elif text == "epsg":
            return f"epsg:{epsg}"

    @staticmethod
    def getproj4(epsg):
        """
        Gets projection file (.prj) text for given epsg code from
        spatialreference.org. See: https://www.epsg-registry.org/
        Parameters
        ----------
        epsg : int
            epsg code for coordinate system
        Returns
        -------
        prj : str
            text for a projection (*.prj) file.
        """
        return CRS.get_spatialreference(epsg, text="proj4")


class EpsgReference:
    """
    Sets up a local database of text representations of coordinate reference
    systems, keyed by EPSG code.
    """

    def __init__(self):
        import os
        try:
            from appdirs import user_data_dir
        except ImportError:
            user_data_dir = None
        if user_data_dir:
            datadir = user_data_dir("pymarthe")
        else:
            # if appdirs is not installed, use user's home directory
            datadir = os.path.join(os.path.expanduser("~"), ".pymarthe")
        if not os.path.isdir(datadir):
            os.makedirs(datadir)
        dbname = "epsgref.json"
        self.location = os.path.join(datadir, dbname)


    def to_dict(self):
        """
        returns dict with EPSG code integer key, and WKT CRS text
        """
        import os
        import json
        data = {}
        
        if os.path.exists(self.location):
            with open(self.location, "r") as f:
                loaded_data = json.load(f)
            # convert JSON key from str to EPSG integer
            for key, value in loaded_data.items():
                try:
                    data[int(key)] = value
                except ValueError:
                    data[key] = value
        return data

    def _write(self, data):
        import json
        with open(self.location, "w") as f:
            json.dump(data, f, indent=0)
            f.write("\n")

    def reset(self, verbose=True):
        if os.path.exists(self.location):
            if verbose:
                print(f"Resetting {self.location}")
            os.remove(self.location)
        elif verbose:
            print(f"{self.location} does not exist, no reset required")

    def add(self, epsg, prj):
        """
        add an epsg code to epsgref.json
        """
        data = self.to_dict()
        data[epsg] = prj
        self._write(data)

    def get(self, epsg):
        """
        returns prj from a epsg code, otherwise None if not found
        """
        data = self.to_dict()
        return data.get(epsg)

    def remove(self, epsg):
        """
        removes an epsg entry from epsgref.json
        """
        data = self.to_dict()
        if epsg in data:
            del data[epsg]
            self._write(data)

    @staticmethod
    def show():
        ep = EpsgReference()
        prj = ep.to_dict()
        for k, v in prj.items():
            print(f"{k}:\n{v}\n")


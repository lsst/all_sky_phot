from lsst.all_sky_phot.netbpmfile import NetpbmFile
import subprocess
import re
import datetime
import os
from astropy.time import Time


__all__ = ['readcr2']


def readcr2(filename):
    """
    Read a Canon raw cr2 file. Requires dcraw (which I think is on Macs by default)

    Parameters
    ----------
    filename : str
       File to read.

    Returns:
    im_ppm : numpy.array
        Numpy array of the image data. Indexed by 0=R 1=G 2=B.
    header : dict
        A dictionary of metadata scraped from the image file
    """

    header = {}

    # Convert the file to a temporary ppm
    # Converting the CR2 to PPM
    p = subprocess.Popen(["dcraw", "-6", "-j", "-W", filename]).communicate()[0]

    # Getting the EXIF of CR2 with dcraw
    p = subprocess.Popen(["dcraw", "-i", "-v", filename], stdout=subprocess.PIPE)
    cr2header = p.communicate()[0]
    cr2header = cr2header.decode("utf-8")
    # Catching the Timestamp
    m = re.search('(?<=Timestamp:).*', cr2header)
    date1 = m.group(0).split()
    months = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6, 'Jul': 7, 'Aug': 8, 'Sep': 9,
              'Oct': 10, 'Nov': 11, 'Dec': 12}
    date = datetime.datetime(int(date1[4]), months[date1[1]], int(date1[2]),
                             int(date1[3].split(':')[0]), int(date1[3].split(':')[1]),
                             int(date1[3].split(':')[2]))
    time = Time('{0:%Y-%m-%d %H:%M:%S}'.format(date))
    mjd = time.mjd
    header['mjd'] = mjd

    date = '{0:%Y-%m-%d %H:%M:%S}'.format(date)
    header['date'] = date

    # Catching the Shutter Speed
    m = re.search('(?<=Shutter:).*(?=sec)', cr2header)
    header['shutter'] = m.group(0).strip()

    # Catching the Aperture
    m = re.search('(?<=Aperture: f/).*', cr2header)
    header['aperture'] = m.group(0).strip()

    # Catching the ISO Speed
    m = re.search('(?<=ISO speed:).*', cr2header)
    header['iso'] = m.group(0).strip()

    # Catching the Focal length
    m = re.search('(?<=Focal length: ).*(?=mm)', cr2header)
    header['focal'] = m.group(0).strip()

    # Catching the Original Filename of the cr2
    m = re.search('(?<=Filename:).*', cr2header)
    header['original_file'] = m.group(0).strip()

    # Catching the Camera Type
    m = re.search('(?<=Camera:).*', cr2header)
    header['camera'] = m.group(0).strip()

    ppm_name = filename[:-4] + '.ppm'
    im_ppm = NetpbmFile(ppm_name).asarray()
    # Delete the temp ppm file
    os.unlink(ppm_name)

    return im_ppm, header

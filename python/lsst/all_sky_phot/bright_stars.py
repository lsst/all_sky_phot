
from __future__ import print_function
from lsst.sims.catUtils.baseCatalogModels import BrightStarObj

# if on UW campus/VPN
db = BrightStarObj(database='LSSTCATSIM', host='fatboy.phys.washington.edu',
                   port=1433, driver='mssql+pymssql')

# if using ssh tunnel
#db = BrightStarObj()

column_names = db.get_column_names(tableName='bright_stars')
print("Available columns:")
for name in column_names:
    print(name)

# columns to query
col_names = ['ra', 'decl', 'umag', 'gmag', 'rmag', 'imag', 'zmag', 'ymag']

# return the data in chunks of 10000 objects at a time

chunk_size = 10000
limit = 40000
data = db.query_columns(colnames=col_names, constraint='rmag<6.0',
                        chunk_size=chunk_size, limit=limit)

# note the limit kwarg limits this example to returning 20000 objects

# you will need to iterate over those chunks of 10000 objects;
# the actuall stars can be accessed one at a time with something like
#
# for chunk in data:
#    for star in chunk:
#        do something to star
#
# note that, whatever columns you ask for, query_columns automatically
# prepends the id columns (simobjid) to each row.
#stars = []
#for chunk in data:
#    for star in chunk:
#        stars.append(star)
chunks = []
for chunk in data:
    chunks.append(chunk)
